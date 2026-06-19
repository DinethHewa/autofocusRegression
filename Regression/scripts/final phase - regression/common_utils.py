#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def exists_all(paths: list[str]) -> bool:
    return all(Path(p).exists() for p in paths)


def run_subprocess(cmd: list[str]) -> None:
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def safe_write_csv(df, path: str, force: bool = False) -> None:
    out_path = Path(path)
    if out_path.exists() and not force:
        print(f"[WARN] Skip existing file (use --force to overwrite): {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DONE] Wrote: {out_path}")
