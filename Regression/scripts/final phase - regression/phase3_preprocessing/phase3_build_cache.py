#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common_paths import manifest_path
from common_utils import ensure_dir
from config_phase3 import TRACK_DATASETS, build_phase3_config
from preprocess_ops import (
    assemble_XA_XB,
    compute_dog_lite,
    compute_dwt_ehf,
    percentile_clip,
    rescale01,
    to_grayscale,
    zscore,
)
from tiling import cut_to_grid

WORKER_CTX: dict[str, Any] = {}
MIN_FREE_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024  # 2 GiB


def _load_image(path: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    if cv2 is not None:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

    with Image.open(p) as im:
        return np.asarray(im)


def _is_roi_image(img: np.ndarray, roi_size: int) -> bool:
    return img.shape[0] == roi_size and img.shape[1] == roi_size


def _prep_I(patch: np.ndarray, cfg_dict: dict[str, Any]) -> np.ndarray:
    I = to_grayscale(patch)
    I = percentile_clip(I, p_low=float(cfg_dict["p_low"]), p_high=float(cfg_dict["p_high"]))
    I = rescale01(I, eps=float(cfg_dict["eps"]))
    I = zscore(I, eps=float(cfg_dict["eps"]))
    return I.astype(np.float32)


def _dtype_from_name(name: str):
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(f"Unsupported cache dtype: {name}")


def _cache_uid(track: str, dataset: str, source_image_path: str, patch_id: str, cfg_dict: dict[str, Any]) -> str:
    key = "|".join(
        [
            track,
            dataset,
            source_image_path,
            patch_id,
            str(tuple(cfg_dict["sigmas"])),
            str(cfg_dict["roi_size"]),
            str(cfg_dict["wavelet"]),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


def _init_stats() -> dict[str, Any]:
    return {
        "roi_count": 0,
        "dog_time_total_s": 0.0,
        "dwt_time_total_s": 0.0,
        "sum_ch": np.zeros(4, dtype=np.float64),
        "sum_sq_ch": np.zeros(4, dtype=np.float64),
        "pixel_count": 0,
    }


def _accumulate_stats(stats: dict[str, Any], xb: np.ndarray, dog_time_s: float, dwt_time_s: float) -> None:
    flat = xb.reshape(-1, 4).astype(np.float64)
    stats["sum_ch"] += flat.sum(axis=0)
    stats["sum_sq_ch"] += np.square(flat).sum(axis=0)
    stats["pixel_count"] += flat.shape[0]
    stats["roi_count"] += 1
    stats["dog_time_total_s"] += dog_time_s
    stats["dwt_time_total_s"] += dwt_time_s


def _safe_meta(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass
    try:
        # Use an explicit file handle so numpy does not append an extra ".npy" suffix.
        with tmp_path.open("wb") as f:
            np.save(f, arr, allow_pickle=False)
        os.replace(tmp_path, path)
    except OSError as exc:
        for p in [tmp_path, path]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        usage = shutil.disk_usage(path.parent)
        free_gb = usage.free / (1024 ** 3)
        raise OSError(
            f"Failed writing cache array to {path}. Free space in target filesystem: {free_gb:.2f} GiB. "
            f"Original error: {exc}"
        ) from exc


def _ensure_min_free_space(cache_dir: Path, min_free_bytes: int = MIN_FREE_BYTES_DEFAULT) -> None:
    usage = shutil.disk_usage(cache_dir)
    if usage.free < int(min_free_bytes):
        free_gb = usage.free / (1024 ** 3)
        need_gb = min_free_bytes / (1024 ** 3)
        raise OSError(
            f"Insufficient free space for cache build at {cache_dir}. "
            f"Free={free_gb:.2f} GiB, required>={need_gb:.2f} GiB."
        )


def _process_single_patch(
    row: dict[str, Any],
    patch_id: str,
    patch: np.ndarray,
    cfg_dict: dict[str, Any],
    cache_dir: Path,
    force: bool,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    roi_size = int(cfg_dict["roi_size"])
    if patch.shape[0] != roi_size or patch.shape[1] != roi_size:
        raise ValueError(f"ROI must be {roi_size}x{roi_size}; got {patch.shape} for patch_id={patch_id}")

    track = str(cfg_dict["track"])
    dataset = str(row["dataset"])
    source_path = str(row["image_path"])

    uid = _cache_uid(track, dataset, source_path, patch_id, cfg_dict)
    ds_dir = cache_dir / dataset
    ensure_dir(str(ds_dir))

    xa_path = ds_dir / f"{uid}_XA.npy"
    xb_path = ds_dir / f"{uid}_XB.npy"
    meta_path = ds_dir / f"{uid}_meta.json"

    record = {
        "roi_uid": uid,
        "source_image_path": source_path,
        "patch_id": patch_id,
        "cache_path_XA": str(xa_path),
        "cache_path_XB": str(xb_path),
        "track": track,
        "dataset": dataset,
        "status": "pending",
    }

    for k in ["slide_id", "patient_id", "stack_id", "z_index", "z_best", "defocus_um", "delta_z", "y_sign", "y_mag"]:
        if k in row:
            record[k] = row[k]

    stats = _init_stats()

    if xa_path.exists() and xb_path.exists() and meta_path.exists() and not force:
        record["status"] = "skipped_exists"
        return record, stats

    if dry_run:
        record["status"] = "dry_run"
        return record, stats

    I = _prep_I(patch, cfg_dict)

    t0 = time.perf_counter()
    D1, D2 = compute_dog_lite(I, tuple(cfg_dict["sigmas"]), eps=float(cfg_dict["eps"]))
    dog_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    EHF = compute_dwt_ehf(
        I,
        wavelet=str(cfg_dict["wavelet"]),
        roi_size=int(cfg_dict["roi_size"]),
        eps=float(cfg_dict["eps"]),
    )
    dwt_time = time.perf_counter() - t1

    XA, XB = assemble_XA_XB(I, D1, D2, EHF, roi_size=roi_size)

    dtype_cache = _dtype_from_name(str(cfg_dict["dtype_cache"]))
    _atomic_save_npy(xa_path, XA.astype(dtype_cache, copy=False))
    _atomic_save_npy(xb_path, XB.astype(dtype_cache, copy=False))

    with meta_path.open("w", encoding="utf-8") as f:
        meta = _safe_meta(row)
        meta.update({"roi_uid": uid, "patch_id": patch_id, "track": track, "dataset": dataset})
        json.dump(meta, f, indent=2)

    _accumulate_stats(stats, XB, dog_time, dwt_time)
    record["status"] = "written"
    return record, stats


def _process_manifest_row(row: dict[str, Any], cfg_dict: dict[str, Any], cache_dir: Path, force: bool, dry_run: bool):
    img = _load_image(str(row["image_path"]))
    roi_size = int(cfg_dict["roi_size"])

    if _is_roi_image(img, roi_size=roi_size):
        patches = [("r00_c00", img)]
    else:
        patches = cut_to_grid(img, grid=(10, 10), roi_size=roi_size)

    out_records = []
    out_stats = _init_stats()
    for patch_id, patch in patches:
        rec, st = _process_single_patch(row, patch_id, patch, cfg_dict, cache_dir, force, dry_run)
        out_records.append(rec)
        out_stats["roi_count"] += st["roi_count"]
        out_stats["dog_time_total_s"] += st["dog_time_total_s"]
        out_stats["dwt_time_total_s"] += st["dwt_time_total_s"]
        out_stats["sum_ch"] += st["sum_ch"]
        out_stats["sum_sq_ch"] += st["sum_sq_ch"]
        out_stats["pixel_count"] += st["pixel_count"]

    return out_records, out_stats


def _worker_init(cfg_dict: dict[str, Any], cache_dir: str, force: bool, dry_run: bool) -> None:
    WORKER_CTX["cfg_dict"] = cfg_dict
    WORKER_CTX["cache_dir"] = Path(cache_dir)
    WORKER_CTX["force"] = force
    WORKER_CTX["dry_run"] = dry_run


def _worker_fn(row: dict[str, Any]):
    return _process_manifest_row(
        row,
        cfg_dict=WORKER_CTX["cfg_dict"],
        cache_dir=WORKER_CTX["cache_dir"],
        force=bool(WORKER_CTX["force"]),
        dry_run=bool(WORKER_CTX["dry_run"]),
    )


def _merge_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    merged = _init_stats()
    for st in stats_list:
        merged["roi_count"] += st["roi_count"]
        merged["dog_time_total_s"] += st["dog_time_total_s"]
        merged["dwt_time_total_s"] += st["dwt_time_total_s"]
        merged["sum_ch"] += st["sum_ch"]
        merged["sum_sq_ch"] += st["sum_sq_ch"]
        merged["pixel_count"] += st["pixel_count"]
    return merged


def _stats_to_df(stats: dict[str, Any]) -> pd.DataFrame:
    rows = []
    ch_names = ["I", "D1", "D2", "E_HF"]
    if stats["pixel_count"] > 0:
        means = stats["sum_ch"] / stats["pixel_count"]
        vars_ = (stats["sum_sq_ch"] / stats["pixel_count"]) - np.square(means)
        stds = np.sqrt(np.maximum(vars_, 0.0))
    else:
        means = np.zeros(4, dtype=np.float64)
        stds = np.zeros(4, dtype=np.float64)

    for i, name in enumerate(ch_names):
        rows.append({"metric": f"channel_mean_{name}", "value": float(means[i])})
        rows.append({"metric": f"channel_std_{name}", "value": float(stds[i])})

    roi_count = max(int(stats["roi_count"]), 1)
    rows.append({"metric": "roi_count", "value": float(stats["roi_count"])})
    rows.append({"metric": "dog_time_per_roi_ms", "value": 1000.0 * float(stats["dog_time_total_s"]) / roi_count})
    rows.append({"metric": "dwt_time_per_roi_ms", "value": 1000.0 * float(stats["dwt_time_total_s"]) / roi_count})
    rows.append({"metric": "total_dog_time_s", "value": float(stats["dog_time_total_s"])})
    rows.append({"metric": "total_dwt_time_s", "value": float(stats["dwt_time_total_s"])})
    return pd.DataFrame(rows)


def _default_manifest_paths(track: str, datasets: list[str]) -> list[Path]:
    paths = [Path(manifest_path(ds)) for ds in datasets]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing manifest files: {missing}")
    return paths


def _infer_dataset_from_manifest(manifest_p: Path) -> str:
    stem = manifest_p.stem
    if stem.startswith("manifest_"):
        return stem[len("manifest_") :]
    raise ValueError(f"Cannot infer dataset from manifest path: {manifest_p}")


def _load_manifest_rows(manifest_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for mpth in manifest_paths:
        if not mpth.is_file():
            raise FileNotFoundError(f"Manifest not found: {mpth}")
        df = pd.read_csv(mpth)
        if "image_path" not in df.columns:
            raise ValueError(f"Manifest missing required column 'image_path': {mpth}")
        if "dataset" not in df.columns:
            df["dataset"] = _infer_dataset_from_manifest(mpth)
        else:
            inferred = _infer_dataset_from_manifest(mpth)
            df["dataset"] = df["dataset"].fillna(inferred).replace("", inferred)
        frames.append(df)

    if not frames:
        raise ValueError("No manifest rows loaded")
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["image_path"]).reset_index(drop=True)
    out["image_path"] = out["image_path"].astype(str)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase-3 preprocessing cache (XA/XB) for one track.")
    parser.add_argument("--track", required=True, choices=["smears", "biopsy"])
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset list within track")
    parser.add_argument("--manifest-paths", nargs="+", default=None, help="Optional explicit manifest CSV paths")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Process only first N manifest rows")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed used when --max-samples is set")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker processes (default: 1)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache entries")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without writing cache")
    parser.add_argument("--resume", action="store_true", help="Reuse existing cache index/stats and skip when complete.")
    args = parser.parse_args()

    cfg = build_phase3_config(args.track)
    datasets = TRACK_DATASETS[args.track] if args.datasets is None else list(dict.fromkeys(args.datasets))

    allowed = set(TRACK_DATASETS[args.track])
    if not set(datasets).issubset(allowed):
        raise ValueError(f"Invalid datasets for track={args.track}. Allowed={sorted(allowed)}")

    if args.manifest_paths is not None:
        manifest_paths = [Path(p) for p in args.manifest_paths]
    else:
        manifest_paths = _default_manifest_paths(args.track, datasets)

    cache_dir = Path(args.cache_dir) if args.cache_dir else cfg.cache_dir_default
    if cache_dir.resolve() != cfg.cache_dir_default.resolve():
        print(f"[WARN] Using non-default cache dir override: {cache_dir}")
    ensure_dir(str(cache_dir))
    _ensure_min_free_space(cache_dir)

    out_base = cfg.out_dir_default
    metrics_dir = out_base / "metrics"
    ensure_dir(str(out_base))
    ensure_dir(str(metrics_dir))
    cache_index_path = cache_dir / "cache_index.csv"
    stats_path = metrics_dir / "preprocessing_stats.csv"
    config_path = out_base / "preprocessing_config.json"

    if args.resume and (not args.force) and (not args.dry_run):
        if cache_index_path.is_file() and stats_path.is_file() and config_path.is_file():
            print(f"[INFO] Resume: Phase-3 cache outputs already exist for track={args.track}; skipping.")
            print(f"[INFO] Existing index: {cache_index_path}")
            return

    cfg_payload = cfg.to_dict()
    cfg_payload.update(
        {
            "datasets": datasets,
            "manifest_paths": [str(p) for p in manifest_paths],
            "cache_dir": str(cache_dir),
            "max_samples": args.max_samples,
            "sample_seed": args.sample_seed,
            "num_workers": args.num_workers,
            "force": args.force,
            "dry_run": args.dry_run,
        }
    )
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg_payload, f, indent=2)
    print(f"[DONE] Wrote preprocessing config: {config_path}")

    all_rows = _load_manifest_rows(manifest_paths)
    all_rows = all_rows[all_rows["dataset"].isin(datasets)].reset_index(drop=True)
    if all_rows.empty:
        raise ValueError("No manifest rows found for requested datasets.")

    if args.max_samples is not None:
        n = min(int(args.max_samples), len(all_rows))
        all_rows = all_rows.sample(n=n, random_state=int(args.sample_seed)).reset_index(drop=True)
        print(f"[INFO] Random-sampled manifest rows: {len(all_rows)} (seed={int(args.sample_seed)})")

    records: list[dict[str, Any]] = []
    stats_list: list[dict[str, Any]] = []

    cfg_dict = cfg.to_dict()
    source_records = all_rows.to_dict(orient="records")
    n_workers = max(1, int(args.num_workers))

    start = time.perf_counter()
    if n_workers == 1:
        for i, row in enumerate(source_records, start=1):
            recs, st = _process_manifest_row(row, cfg_dict, cache_dir, args.force, args.dry_run)
            records.extend(recs)
            stats_list.append(st)
            if i % 50 == 0:
                print(f"[INFO] Processed manifest rows: {i}/{len(source_records)}")
    else:
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(cfg_dict, str(cache_dir), args.force, args.dry_run),
        ) as pool:
            for i, (recs, st) in enumerate(pool.imap_unordered(_worker_fn, source_records), start=1):
                records.extend(recs)
                stats_list.append(st)
                if i % 50 == 0:
                    print(f"[INFO] Processed manifest rows: {i}/{len(source_records)}")

    elapsed = time.perf_counter() - start

    index_df = pd.DataFrame(records)
    if index_df.empty:
        raise RuntimeError("No cache records produced.")

    cache_index_path = cache_dir / "cache_index.csv"
    index_df.to_csv(cache_index_path, index=False)
    print(f"[DONE] Wrote cache index: {cache_index_path} rows={len(index_df)}")

    stats = _merge_stats(stats_list)
    stats_df = _stats_to_df(stats)
    stats_df = pd.concat(
        [
            stats_df,
            pd.DataFrame(
                [
                    {"metric": "total_elapsed_s", "value": float(elapsed)},
                    {"metric": "manifest_rows", "value": float(len(source_records))},
                ]
            ),
        ],
        ignore_index=True,
    )
    stats_path = metrics_dir / "preprocessing_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[DONE] Wrote preprocessing stats: {stats_path}")


if __name__ == "__main__":
    main()
