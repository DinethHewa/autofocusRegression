#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    build_phase5_index_from_sources,
    default_cache_index_path,
    ensure_reg_output_tree,
    load_cache_index,
    resolve_reg_out_dir,
    save_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build unified Phase-5 regression index from manifest defocus_um + cache index.")
    ap.add_argument("--track", required=True, choices=["smears", "biopsy"])
    ap.add_argument("--cache-index", default=None, help="Path to cache_phase3/<track>/cache_index.csv")
    ap.add_argument("--out-dir", default=None, help="Output dir (must be under out_final_phase/<track>/regression)")
    ap.add_argument("--resume", action="store_true", help="Reuse existing index outputs when present.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_reg_out_dir(args.track, args.out_dir)
    paths = ensure_reg_output_tree(out_dir)
    out_csv = paths["root"] / "index_phase5.csv"
    cfg_path = paths["root"] / "index_phase5_config.json"

    if args.resume and out_csv.is_file() and cfg_path.is_file():
        print(f"[INFO] Resume: Phase5 index already exists at {out_csv}; skipping.")
        return

    cache_index = str(default_cache_index_path(args.track) if args.cache_index is None else args.cache_index)
    cache_df = load_cache_index(track=args.track, cache_index_path=cache_index)
    index_df = build_phase5_index_from_sources(track=args.track, cache_df=cache_df)

    index_df.to_csv(out_csv, index=False)

    cfg = {
        "track": args.track,
        "cache_index": cache_index,
        "rows": int(len(index_df)),
        "datasets": sorted(index_df["dataset"].astype(str).unique().tolist()),
        "out_csv": str(out_csv),
    }
    save_json(cfg, cfg_path)

    print(f"[DONE] Wrote Phase5 index: {out_csv} rows={len(index_df)}")


if __name__ == "__main__":
    main()
