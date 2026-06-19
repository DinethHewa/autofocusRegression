#!/usr/bin/env python3
"""
Phase-3 manifest creation runner.

Creates manifests for:
- BMA
- PBS
- WBC
- focus_train
- focus_test

Does NOT create TBF (intentionally removed for Phase-3).

Example:
  python create_manifests_phase3.py --bma --pbs --wbc
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(script: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script)] + args
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase-3 manifest creation wrappers.")
    ap.add_argument("--bma", action="store_true")
    ap.add_argument("--pbs", action="store_true")
    ap.add_argument("--wbc", action="store_true")
    ap.add_argument("--focus-train", action="store_true")
    ap.add_argument("--focus-test", action="store_true")
    ap.add_argument("--use-gpu", action="store_true", help="Forwarded to stack-based scripts.")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    selected_any = any([args.bma, args.pbs, args.wbc, args.focus_train, args.focus_test])
    if not selected_any:
        args.bma = args.pbs = args.wbc = args.focus_train = args.focus_test = True

    if args.bma:
        _run(here / "manifest_creation_bma.py", ["--use-gpu"] if args.use_gpu else [])
    if args.pbs:
        _run(here / "manifest_creation_pbs.py", ["--use-gpu"] if args.use_gpu else [])
    if args.wbc:
        _run(here / "manifest_creation_wbc.py", ["--use-gpu"] if args.use_gpu else [])
    if args.focus_train:
        _run(here / "manifest_creation_focus_train.py", [])
    if args.focus_test:
        _run(here / "manifest_creation_focus_test.py", [])


if __name__ == "__main__":
    main()

