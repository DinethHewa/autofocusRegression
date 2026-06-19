#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR / 'evaluation' / 'build_q1_paper_package.py'


if __name__ == '__main__':
    cmd = [sys.executable, str(TARGET), *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)
