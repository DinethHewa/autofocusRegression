#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from common_paths import PHASE3_MANIFEST_CREATION_DIR, manifest_path, table_path
from common_utils import ensure_dir, exists_all, run_subprocess, safe_write_csv

GROUP_DATASETS = ["pbs", "wbc", "bma"]
PHASE3_FLAG_BY_DATASET = {
    "pbs": "--pbs",
    "wbc": "--wbc",
    "bma": "--bma",
}


def _phase3_runner() -> Path:
    return Path(PHASE3_MANIFEST_CREATION_DIR) / "create_manifests_phase3.py"


def _phase3_dataset_script(dataset: str) -> Path:
    return Path(PHASE3_MANIFEST_CREATION_DIR) / f"manifest_creation_{dataset}.py"


def _fail_missing_phase3(dataset: str) -> None:
    runner = _phase3_runner()
    creator = _phase3_dataset_script(dataset)
    print("[WARN] Missing required Phase-3 manifest script(s).")
    print(f"[WARN] Expected runner: {runner}")
    print(f"[WARN] Expected creator for '{dataset}': {creator}")
    raise SystemExit(1)


def _expected_outputs(dataset: str) -> tuple[str, str]:
    return manifest_path(dataset), table_path(dataset)


def _find_existing_table_candidate(dataset: str) -> Path | None:
    expected = Path(table_path(dataset))
    dataset_dir = expected.parent
    if not dataset_dir.is_dir():
        return None
    csvs = sorted([p for p in dataset_dir.glob("*.csv") if p.is_file()])
    if not csvs:
        return None
    preferred = [p for p in csvs if dataset in p.name.lower() and "table" in p.name.lower()]
    return preferred[0] if preferred else csvs[0]


def _ensure_manifest_and_table(dataset: str, force: bool) -> bool:
    man_path_str, table_path_str = _expected_outputs(dataset)
    man_path = Path(man_path_str)
    std_table = Path(table_path_str)
    ensure_dir(str(std_table.parent))

    if not man_path.exists():
        print(f"[WARN] Missing manifest after generation: {man_path}")
        return False

    if std_table.exists() and not force:
        return True

    if std_table.exists() and force:
        print(f"[INFO] Overwriting table due to --force: {std_table}")

    candidate = _find_existing_table_candidate(dataset)
    if candidate is not None and candidate.resolve() != std_table.resolve():
        if std_table.exists() and not force:
            print(f"[WARN] Standard table exists; keeping: {std_table}")
            return True
        shutil.copy2(candidate, std_table)
        print(f"[INFO] Copied table to standard name: {candidate} -> {std_table}")
        return True

    try:
        df = pd.read_csv(man_path)
    except Exception as exc:
        print(f"[WARN] Could not read manifest for table fallback ({man_path}): {exc}")
        return False

    safe_write_csv(df, str(std_table), force=force)
    return std_table.exists()


def _should_skip_dataset(dataset: str, force: bool) -> bool:
    if force:
        return False
    man_path, tab_path = _expected_outputs(dataset)
    return exists_all([man_path, tab_path])


def _run_dataset_creator(dataset: str, dry_run: bool) -> None:
    runner = _phase3_runner()
    creator = _phase3_dataset_script(dataset)

    if runner.is_file():
        cmd = [sys.executable, str(runner), PHASE3_FLAG_BY_DATASET[dataset]]
    elif creator.is_file():
        cmd = [sys.executable, str(creator)]
    else:
        _fail_missing_phase3(dataset)

    if dry_run:
        print(f"[INFO] Dry-run: {' '.join(cmd)}")
        return

    run_subprocess(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create/reuse manifests and tables for smears datasets: pbs, wbc, bma."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate and overwrite standard outputs even if they already exist.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=GROUP_DATASETS,
        default=GROUP_DATASETS,
        help="Subset of smears datasets to process (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without running Phase-3 creators or writing files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume-safe mode. Existing dataset outputs are reused (default behavior).",
    )
    args = parser.parse_args()
    if args.force and args.resume:
        raise ValueError("--force and --resume cannot be used together.")

    datasets = list(dict.fromkeys(args.datasets))
    failures = []

    for ds in datasets:
        man_path, tab_path = _expected_outputs(ds)
        if _should_skip_dataset(ds, args.force):
            print(f"[INFO] Skip existing outputs for {ds}: {man_path} | {tab_path}")
            continue

        if (not args.force) and Path(man_path).exists() and (not Path(tab_path).exists()):
            print(f"[INFO] Repairing missing table for {ds} from existing outputs.")
            if _ensure_manifest_and_table(ds, force=False):
                print(f"[DONE] Ready: {man_path}")
                print(f"[DONE] Ready: {tab_path}")
                continue

        print(f"[INFO] Processing dataset: {ds}")
        _run_dataset_creator(ds, args.dry_run)

        if args.dry_run:
            print(f"[INFO] Dry-run expected outputs: {man_path} | {tab_path}")
            continue

        ok = _ensure_manifest_and_table(ds, args.force)
        if ok:
            print(f"[DONE] Ready: {man_path}")
            print(f"[DONE] Ready: {tab_path}")
        else:
            failures.append(ds)

    if failures:
        print(f"[WARN] Failed datasets: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
