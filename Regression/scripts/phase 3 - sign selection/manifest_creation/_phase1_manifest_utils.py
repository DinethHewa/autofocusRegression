#!/usr/bin/env python3
"""
Utilities for Phase-3 manifest creation wrappers.

These wrappers reuse the Phase-1 manifest scripts (focus-measure voting, etc.)
but post-process outputs to enforce a consistent defocus convention:
  underfocused (negative) -> focused (0) -> overfocused (positive)

Notes:
- Phase-1 stack-based scripts assign defocus as (focus_idx - i) * step, which is
  the opposite sign of the "under->over" convention used elsewhere in Phase-3.
  We fix this by flipping the sign in the output tables/manifests.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def load_phase1_module(phase1_script_filename: str) -> ModuleType:
    """
    Load a Phase-1 script as a Python module by absolute path.
    This avoids issues with spaces in folder names.
    """
    here = Path(__file__).resolve()
    scripts_dir = here.parents[2]  # .../journal/Regression/scripts
    phase1_dir = scripts_dir / "phase 1- manifest creation"
    script_path = phase1_dir / phase1_script_filename
    if not script_path.is_file():
        raise FileNotFoundError(f"Phase-1 script not found: {script_path}")

    mod_name = f"phase1_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def flip_defocus_sign(df: pd.DataFrame, defocus_col: str = "defocus_um") -> pd.DataFrame:
    out = df.copy()
    out[defocus_col] = -pd.to_numeric(out[defocus_col], errors="coerce")
    return out


def sort_under_focus_over(
    df: pd.DataFrame,
    sort_cols: List[str],
) -> pd.DataFrame:
    """
    Sort so that negative defocus comes first, then 0, then positive.
    If sort_cols includes defocus_um, a simple ascending sort achieves this.
    """
    out = df.copy()
    # Ensure stable, numeric sort for defocus if present.
    if "defocus_um" in sort_cols and "defocus_um" in out.columns:
        out["defocus_um"] = pd.to_numeric(out["defocus_um"], errors="coerce")
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def drop_first_per_group(
    df: pd.DataFrame,
    group_col: str,
    already_sorted: bool = False,
    sort_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Drop the first row in each group (after sorting).
    Used to remove the most underfocused slice per stack for BMA/PBS.
    """
    if not already_sorted:
        if not sort_cols:
            raise ValueError("sort_cols must be provided when already_sorted=False")
        df = sort_under_focus_over(df, sort_cols=sort_cols)

    out = df.copy()
    out["_row_in_group"] = out.groupby(group_col).cumcount()
    out = out[out["_row_in_group"] > 0].drop(columns=["_row_in_group"]).reset_index(drop=True)
    return out


def write_manifest_csv(df: pd.DataFrame, out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

