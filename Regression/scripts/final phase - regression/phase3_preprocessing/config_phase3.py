#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from common_paths import DATA_DIR


TRACK_SIGMAS = {
    "smears": (0.7, 1.4, 2.8),
    "biopsy": (1.0, 2.0, 4.0),
}

TRACK_DATASETS = {
    "smears": ["pbs", "wbc", "bma"],
    "biopsy": ["focus_train", "focus_test"],
}


@dataclass
class Phase3Config:
    track: str
    sigmas: tuple[float, float, float]
    roi_size: int = 200
    p_low: float = 1.0
    p_high: float = 99.0
    eps: float = 1e-6
    wavelet: str = "haar"
    dwt_level: int = 1
    upsample_mode: str = "bilinear"
    dtype_cache: str = "float16"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cache_dir_default"] = str(self.cache_dir_default)
        d["out_dir_default"] = str(self.out_dir_default)
        return d

    @property
    def cache_dir_default(self) -> Path:
        return Path(DATA_DIR) / "cache_phase3" / self.track

    @property
    def out_dir_default(self) -> Path:
        return Path(DATA_DIR) / "out_final_phase" / self.track


def build_phase3_config(track: str) -> Phase3Config:
    if track not in TRACK_SIGMAS:
        raise ValueError(f"Unsupported track: {track}")
    return Phase3Config(track=track, sigmas=TRACK_SIGMAS[track])
