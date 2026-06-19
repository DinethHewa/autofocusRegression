#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from common_paths import DATA_DIR, TABLES_DIR

TRACK_DATASETS = {
    "smears": ["pbs", "wbc", "bma"],
    "biopsy": ["focus_train", "focus_test"],
}


@dataclass
class TrainConfig:
    image_size: int = 224
    sign_epochs: int = 4
    reg_epochs: int = 6
    sign_batch_size: int = 16
    reg_batch_size: int = 16
    sign_lr: float = 1e-3
    reg_lr: float = 1e-4
    huber_delta: float = 0.5
    triplet_margin: float = 0.2
    triplet_weight: float = 0.1
    triplet_bin_width: float = 0.5
    wrong_direction_target: float = 0.05
    vote_early_margin: float = 0.6
    min_rows_per_branch: int = 16


@dataclass
class FinalPhaseConfig:
    track: str
    datasets: list[str]
    k_rois: int = 8
    min_rois_per_fov: int = 2
    sigma1: float = 0.8
    sigma2: float = 1.6
    sigma3: float = 2.4
    tau_init: float = 0.6
    split_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def out_base_dir(self) -> Path:
        return Path(DATA_DIR) / "out_final_phase" / self.track

    @property
    def out_manifests_dir(self) -> Path:
        return self.out_base_dir / "manifests"

    @property
    def out_splits_dir(self) -> Path:
        return self.out_base_dir / "splits"

    @property
    def out_models_sign_dir(self) -> Path:
        return self.out_base_dir / "models" / "sign"

    @property
    def out_models_reg_plus_dir(self) -> Path:
        return self.out_base_dir / "models" / "regress_plus"

    @property
    def out_models_reg_minus_dir(self) -> Path:
        return self.out_base_dir / "models" / "regress_minus"

    @property
    def out_metrics_dir(self) -> Path:
        return self.out_base_dir / "metrics"

    @property
    def out_logs_dir(self) -> Path:
        return self.out_base_dir / "logs"

    @property
    def table_summary_dir(self) -> Path:
        return Path(TABLES_DIR) / "FINAL_PHASE" / self.track.upper()

    def ensure_output_dirs(self) -> None:
        for d in [
            self.out_manifests_dir,
            self.out_splits_dir,
            self.out_models_sign_dir,
            self.out_models_reg_plus_dir,
            self.out_models_reg_minus_dir,
            self.out_metrics_dir,
            self.out_logs_dir,
            self.table_summary_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["out_base_dir"] = str(self.out_base_dir)
        out["table_summary_dir"] = str(self.table_summary_dir)
        return out


def make_track_config(track: str, datasets: list[str] | None = None) -> FinalPhaseConfig:
    if track not in TRACK_DATASETS:
        raise ValueError(f"Unsupported track: {track}")
    ds = TRACK_DATASETS[track] if datasets is None else datasets
    return FinalPhaseConfig(track=track, datasets=list(ds))
