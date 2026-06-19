#!/usr/bin/env python3
"""
Phase-3 focusTest manifest creator (wrapper around Phase-1 script).

Changes vs Phase-1:
- Sort output rows as underfocused (negative) -> focused (0) -> overfocused (positive),
  which is simply ascending defocus_um.
"""

from __future__ import annotations

import argparse

from _phase1_manifest_utils import load_phase1_module, sort_under_focus_over, write_manifest_csv


def main() -> None:
    p1 = load_phase1_module("manifest_creation_focus_test.py")

    ap = argparse.ArgumentParser(description="Phase-3 focusTest manifest creator (wrapper).")
    ap.add_argument("--flush-every", type=int, default=500, help="Forwarded to Phase-1 script.")
    ap.add_argument("--roots", nargs="+", default=None, help="Override roots (Windows paths allowed).")
    args = ap.parse_args()

    roots = p1.DEFAULT_ROOTS if args.roots is None else args.roots
    norm_roots = [p1.normalize_root_path(r) for r in roots]

    df = p1.build_manifest(roots=norm_roots, flush_every=args.flush_every)
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    df = sort_under_focus_over(df, sort_cols=["defocus_um", "image_path"])

    write_manifest_csv(df[["image_path", "defocus_um"]], p1.MANIFEST_PATH)
    print(f"[INFO] Wrote {p1.MANIFEST_PATH} rows={len(df)}")


if __name__ == "__main__":
    main()

