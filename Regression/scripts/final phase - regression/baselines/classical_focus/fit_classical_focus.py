#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from baseline_classical_focus import ClassicalFocusBaseline


def main() -> None:
    ap = argparse.ArgumentParser(description='Train/select the classical focus baseline')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    baseline = ClassicalFocusBaseline(track=args.track, seed=args.seed, resume=args.resume)
    baseline.fit()
    print(f'[DONE] Classical focus training outputs: {baseline.root}')


if __name__ == '__main__':
    main()
