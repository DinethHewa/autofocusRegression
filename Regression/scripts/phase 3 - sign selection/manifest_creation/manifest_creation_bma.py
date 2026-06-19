#!/usr/bin/env python3
"""
Phase-3 BMA manifest creator (wrapper around Phase-1 script).

Changes vs Phase-1:
- Enforces defocus ordering: underfocused (negative) -> focused (0) -> overfocused (positive).
- Flips defocus sign to match the Phase-3 convention.
- Drops the first slice PER STACK after ordering (removes the most underfocused image).
- Does NOT generate TBF here (handled by omission in Phase-3 manifest creation folder).

Outputs (same paths as Phase-1):
- /home/dineth/focus_measure/journal/Regression/data/manifest_bma.csv
- /home/dineth/focus_measure/journal/Regression/tables/BMA/bma_table.csv
"""

from __future__ import annotations

import argparse

from _phase1_manifest_utils import (
    drop_first_per_group,
    flip_defocus_sign,
    load_phase1_module,
    sort_under_focus_over,
    write_manifest_csv,
)


def main() -> None:
    p1 = load_phase1_module("manifest_creation_bma.py")

    ap = argparse.ArgumentParser(description="Phase-3 BMA manifest creator (wrapper).")
    ap.add_argument("--use-gpu", action="store_true", help="Forwarded to Phase-1 script.")
    ap.add_argument("--flush-every", type=int, default=200, help="Forwarded to Phase-1 script.")
    args = ap.parse_args()

    manifest_df, table_df = p1.build_manifest(use_gpu=args.use_gpu, flush_every=args.flush_every)

    # Flip sign so underfocused is negative, overfocused positive.
    table_df = flip_defocus_sign(table_df, defocus_col="defocus_um")
    table_df = sort_under_focus_over(table_df, sort_cols=["stack_id", "defocus_um", "image_name"])

    before = len(table_df)
    table_df = drop_first_per_group(table_df, group_col="stack_id", already_sorted=True)
    removed = before - len(table_df)

    manifest_df = table_df[["image_path", "defocus_um"]].copy()
    write_manifest_csv(manifest_df, p1.MANIFEST_PATH)
    write_manifest_csv(table_df, p1.TABLE_PATH)

    print(f"[INFO] Wrote {p1.MANIFEST_PATH} rows={len(manifest_df)}")
    print(f"[INFO] Wrote {p1.TABLE_PATH} rows={len(table_df)} (dropped first-per-stack: {removed})")


if __name__ == "__main__":
    main()

