#!/usr/bin/env python3
"""
Phase-3 WBC manifest creator (wrapper around Phase-1 script).

Changes vs Phase-1:
- Enforces defocus ordering: underfocused (negative) -> focused (0) -> overfocused (positive).
- Flips defocus sign to match the Phase-3 convention.
"""

from __future__ import annotations

import argparse

from _phase1_manifest_utils import (
    flip_defocus_sign,
    load_phase1_module,
    sort_under_focus_over,
    write_manifest_csv,
)


def main() -> None:
    p1 = load_phase1_module("manifest_creation_wbc.py")

    ap = argparse.ArgumentParser(description="Phase-3 WBC manifest creator (wrapper).")
    ap.add_argument("--use-gpu", action="store_true", help="Forwarded to Phase-1 script.")
    ap.add_argument("--flush-every", type=int, default=200, help="Forwarded to Phase-1 script.")
    args = ap.parse_args()

    manifest_df, table_df = p1.build_manifest(use_gpu=args.use_gpu, flush_every=args.flush_every)

    table_df = flip_defocus_sign(table_df, defocus_col="defocus_um")
    table_df = sort_under_focus_over(table_df, sort_cols=["stack_id", "defocus_um", "image_name"])

    manifest_df = table_df[["image_path", "defocus_um"]].copy()
    write_manifest_csv(manifest_df, p1.MANIFEST_PATH)
    write_manifest_csv(table_df, p1.TABLE_PATH)

    print(f"[INFO] Wrote {p1.MANIFEST_PATH} rows={len(manifest_df)}")
    print(f"[INFO] Wrote {p1.TABLE_PATH} rows={len(table_df)}")


if __name__ == "__main__":
    main()

