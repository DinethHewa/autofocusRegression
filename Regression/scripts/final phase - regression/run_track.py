#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_utils import run_subprocess
from config import TRACK_DATASETS, make_track_config
from pipeline_steps import (
    add_group_id,
    add_labels,
    load_and_validate_manifests,
    make_group_splits,
    roi_gate,
    write_combined_manifest,
    write_config_json,
    write_splits,
)
from train_regression_joint import train_regressors
from train_sign import train_sign_model


def _run_manifest_create(track: str, datasets: list[str], force: bool, dry_run: bool, resume: bool) -> None:
    if track == "smears":
        script = SCRIPT_DIR / "smears" / "create_manifests_smears.py"
    elif track == "biopsy":
        script = SCRIPT_DIR / "biopsy" / "create_manifests_biopsy.py"
    else:
        raise ValueError(f"Unsupported track: {track}")

    if not script.is_file():
        raise FileNotFoundError(f"Missing manifest runner: {script}")

    cmd = [sys.executable, str(script), "--datasets", *datasets]
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")
    if resume:
        cmd.append("--resume")

    if dry_run:
        print(f"[INFO] Dry-run: {' '.join(cmd)}")
        return

    run_subprocess(cmd)


def _copy_summary_to_table_dir(paths: list[Path], table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if p.is_file():
            dst = table_dir / p.name
            shutil.copy2(p, dst)
            print(f"[DONE] Summary table: {dst}")


def _aggregate_roi_vote(group_df: pd.DataFrame, tau: float, vote_early_margin: float) -> int:
    work = group_df.sort_values("confidence", ascending=False).copy()
    weighted_sum = 0.0
    total_w = 0.0
    for _, row in work.iterrows():
        direction = 1 if row["sign_prob"] >= 0.5 else -1
        conf = float(max(row["sign_prob"], 1.0 - row["sign_prob"]))
        weight = float(row.get("roi_importance", 1.0)) * conf
        if conf < tau:
            continue
        weighted_sum += weight * direction
        total_w += weight
        if total_w > 0:
            margin = abs(weighted_sum) / total_w
            if margin >= vote_early_margin:
                break
    if total_w == 0:
        return 0
    if weighted_sum > 0:
        return 1
    if weighted_sum < 0:
        return -1
    return 0


def _write_end_to_end_metrics(val_df: pd.DataFrame, sign_model_dir: Path, reg_dir: Path, out_metrics_dir: Path, config) -> Path:
    sign_pred_path = sign_model_dir / "val_sign_predictions.csv"
    reg_pred_path = reg_dir / "val_reg_predictions.csv"
    tau_path = sign_model_dir / "tau.json"

    if not sign_pred_path.is_file() or not reg_pred_path.is_file() or not tau_path.is_file():
        raise FileNotFoundError(
            "Missing prediction artifacts for end-to-end metrics. "
            f"Expected: {sign_pred_path}, {reg_pred_path}, {tau_path}"
        )

    sign_pred = pd.read_csv(sign_pred_path)
    reg_pred = pd.read_csv(reg_pred_path)
    with tau_path.open("r", encoding="utf-8") as f:
        tau = float(json.load(f)["tau"])

    work = val_df.copy()
    work = work.merge(sign_pred[["image_path", "sign_prob", "confidence"]], on="image_path", how="left")
    work = work.merge(reg_pred[["image_path", "mag_pred_plus", "mag_pred_minus"]], on="image_path", how="left")

    if work[["sign_prob", "confidence"]].isna().any().any():
        raise ValueError("Missing sign predictions for some validation rows.")

    if "roi_importance" not in work.columns:
        work["roi_importance"] = 1.0

    fov_key = "fov_id" if "fov_id" in work.columns else "group_id"
    group_sign = (
        work.groupby(fov_key)
        .apply(lambda g: _aggregate_roi_vote(g, tau=tau, vote_early_margin=config.train.vote_early_margin))
        .rename("pred_sign_fov")
        .reset_index()
    )

    work = work.merge(group_sign, on=fov_key, how="left")
    work["pred_sign_fov"] = work["pred_sign_fov"].fillna(0).astype(int)

    plus_fallback = float(work["y_mag"].median())
    minus_fallback = float(work["y_mag"].median())
    work["mag_pred_plus"] = work["mag_pred_plus"].fillna(plus_fallback)
    work["mag_pred_minus"] = work["mag_pred_minus"].fillna(minus_fallback)

    work["pred_mag"] = np.where(
        work["pred_sign_fov"] > 0,
        work["mag_pred_plus"],
        np.where(work["pred_sign_fov"] < 0, work["mag_pred_minus"], 0.0),
    )
    work["pred_signed_dz"] = work["pred_sign_fov"] * work["pred_mag"]

    signed_true = work["delta_z"].to_numpy(dtype=float)
    signed_pred = work["pred_signed_dz"].to_numpy(dtype=float)
    abs_err = np.abs(signed_pred - signed_true)

    wrong_dir_mask = (signed_true != 0) & (np.sign(signed_pred) == -np.sign(signed_true))
    catastrophic_wrong_direction_rate = float(wrong_dir_mask.mean())
    abstain_rate = float((work["pred_sign_fov"] == 0).mean())

    metrics = pd.DataFrame(
        [
            {"metric": "mae_signed", "value": float(abs_err.mean()), "count": len(work)},
            {"metric": "rmse_signed", "value": float(np.sqrt(np.mean((signed_pred - signed_true) ** 2))), "count": len(work)},
            {
                "metric": "catastrophic_wrong_direction_rate",
                "value": catastrophic_wrong_direction_rate,
                "count": int((signed_true != 0).sum()),
            },
            {"metric": "abstain_rate", "value": abstain_rate, "count": len(work)},
            {"metric": "tau_used", "value": tau, "count": len(work)},
        ]
    )

    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_metrics_dir / "end_to_end_metrics.csv"
    metrics.to_csv(out_path, index=False)
    print(f"[DONE] End-to-end metrics: {out_path}")
    return out_path


def _validate_track_datasets(track: str, datasets: list[str]) -> list[str]:
    allowed = set(TRACK_DATASETS[track])
    if not set(datasets).issubset(allowed):
        raise ValueError(f"Datasets {datasets} are not valid for track={track}. Allowed: {sorted(allowed)}")
    return list(dict.fromkeys(datasets))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run final cascade + joint-learning pipeline for one track.")
    ap.add_argument("--track", required=True, choices=["smears", "biopsy"], help="Track to run.")
    ap.add_argument("--force", action="store_true", help="Force manifest recreation and overwrite outputs.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions and expected outputs only.")
    ap.add_argument("--resume", action="store_true", help="Reuse existing outputs and skip completed stages.")
    ap.add_argument("--skip-manifest-create", action="store_true", help="Skip calling manifest creation scripts.")
    ap.add_argument("--datasets", nargs="+", default=None, help="Optional dataset override within selected track.")
    args = ap.parse_args()
    if args.force and args.resume:
        raise ValueError("--force and --resume cannot be used together.")

    datasets = TRACK_DATASETS[args.track] if args.datasets is None else args.datasets
    datasets = _validate_track_datasets(args.track, datasets)

    cfg = make_track_config(args.track, datasets=datasets)
    cfg.ensure_output_dirs()
    final_plus = cfg.out_models_reg_plus_dir / "R_plus.keras"
    final_minus = cfg.out_models_reg_minus_dir / "R_minus.keras"
    full_resume_targets = [
        cfg.out_manifests_dir / "combined_group_manifest.csv",
        cfg.out_splits_dir / "train.csv",
        cfg.out_splits_dir / "val.csv",
        cfg.out_splits_dir / "test.csv",
        cfg.out_base_dir / "config.json",
        cfg.out_models_sign_dir / "sign_model.keras",
        cfg.out_models_sign_dir / "tau.json",
        cfg.out_metrics_dir / "sign_metrics.csv",
        cfg.out_metrics_dir / "reg_metrics.csv",
        cfg.out_metrics_dir / "end_to_end_metrics.csv",
        final_plus,
        final_minus,
    ]
    if args.resume and all(p.is_file() for p in full_resume_targets):
        print(f"[INFO] Resume: all track outputs already exist under {cfg.out_base_dir}; skipping.")
        return

    if not args.skip_manifest_create:
        _run_manifest_create(args.track, datasets=datasets, force=args.force, dry_run=args.dry_run, resume=bool(args.resume))

    if args.dry_run:
        print(f"[INFO] Dry-run complete for track={args.track}. Outputs would be under: {cfg.out_base_dir}")
        raise SystemExit(0)

    # Phases 1-3: load, label, group split, ROI gate
    df = load_and_validate_manifests(datasets=datasets)
    df = add_labels(df)
    df = add_group_id(df)
    df = roi_gate(df, k=cfg.k_rois, min_rois_per_fov=cfg.min_rois_per_fov)

    combined_path = cfg.out_manifests_dir / "combined_group_manifest.csv"
    if args.resume and combined_path.is_file():
        print(f"[INFO] Resume: keeping existing combined manifest: {combined_path}")
    else:
        write_combined_manifest(df, combined_path)

    train_df, val_df, test_df = make_group_splits(
        df,
        seed=cfg.split_seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
    )
    split_targets = [cfg.out_splits_dir / "train.csv", cfg.out_splits_dir / "val.csv", cfg.out_splits_dir / "test.csv"]
    if args.resume and all(p.is_file() for p in split_targets):
        print(f"[INFO] Resume: keeping existing split CSVs in {cfg.out_splits_dir}")
    else:
        write_splits(train_df, val_df, test_df, cfg.out_splits_dir)

    config_path = cfg.out_base_dir / "config.json"
    if args.resume and config_path.is_file():
        print(f"[INFO] Resume: keeping existing config: {config_path}")
    else:
        write_config_json(cfg.to_dict(), config_path)

    # Phase 4: sign training
    sign_model_path = cfg.out_models_sign_dir / "sign_model.keras"
    tau_path = cfg.out_models_sign_dir / "tau.json"
    sign_metrics_path = cfg.out_models_sign_dir / "sign_metrics.csv"
    if args.resume and sign_model_path.is_file() and tau_path.is_file() and sign_metrics_path.is_file():
        print(f"[INFO] Resume: reusing Stage-4 artifacts in {cfg.out_models_sign_dir}")
    else:
        sign_model_path_s, tau_path_s, sign_metrics_path_s = train_sign_model(
            train_df=train_df,
            val_df=val_df,
            out_dir=cfg.out_models_sign_dir,
            config=cfg,
        )
        sign_model_path = Path(sign_model_path_s)
        tau_path = Path(tau_path_s)
        sign_metrics_path = Path(sign_metrics_path_s)

    # Phase 5: joint regression training
    expected_plus = cfg.out_models_reg_plus_dir.parent / "R_plus.keras"
    expected_minus = cfg.out_models_reg_plus_dir.parent / "R_minus.keras"
    reg_metrics_path = cfg.out_models_reg_plus_dir.parent / "reg_metrics.csv"
    if args.resume and expected_plus.is_file() and expected_minus.is_file() and reg_metrics_path.is_file():
        print(f"[INFO] Resume: reusing Stage-5 artifacts in {cfg.out_models_reg_plus_dir.parent}")
        r_plus_path = expected_plus
        r_minus_path = expected_minus
    else:
        r_plus_path_s, r_minus_path_s, reg_metrics_path_s = train_regressors(
            train_df=train_df,
            val_df=val_df,
            out_dir=cfg.out_models_reg_plus_dir.parent,
            config=cfg,
        )
        r_plus_path = Path(r_plus_path_s)
        r_minus_path = Path(r_minus_path_s)
        reg_metrics_path = Path(reg_metrics_path_s)

    # Ensure required model filenames and no mixed dirs
    expected_sign = cfg.out_models_sign_dir / "sign_model.keras"

    if Path(sign_model_path) != expected_sign:
        shutil.copy2(sign_model_path, expected_sign)
    if Path(r_plus_path) != expected_plus:
        shutil.copy2(r_plus_path, expected_plus)
    if Path(r_minus_path) != expected_minus:
        shutil.copy2(r_minus_path, expected_minus)

    # Move regressors into dedicated directories
    if (not args.resume) or (not final_plus.is_file()):
        shutil.copy2(expected_plus, final_plus)
    else:
        print(f"[INFO] Resume: keeping existing {final_plus}")
    if (not args.resume) or (not final_minus.is_file()):
        shutil.copy2(expected_minus, final_minus)
    else:
        print(f"[INFO] Resume: keeping existing {final_minus}")

    # Metrics
    sign_metrics_dst = cfg.out_metrics_dir / "sign_metrics.csv"
    reg_metrics_dst = cfg.out_metrics_dir / "reg_metrics.csv"
    if (not args.resume) or (not sign_metrics_dst.is_file()):
        shutil.copy2(sign_metrics_path, sign_metrics_dst)
    else:
        print(f"[INFO] Resume: keeping existing {sign_metrics_dst}")
    if (not args.resume) or (not reg_metrics_dst.is_file()):
        shutil.copy2(reg_metrics_path, reg_metrics_dst)
    else:
        print(f"[INFO] Resume: keeping existing {reg_metrics_dst}")

    e2e_path = cfg.out_metrics_dir / "end_to_end_metrics.csv"
    if args.resume and e2e_path.is_file():
        print(f"[INFO] Resume: keeping existing {e2e_path}")
    else:
        e2e_path = _write_end_to_end_metrics(
            val_df=val_df,
            sign_model_dir=cfg.out_models_sign_dir,
            reg_dir=cfg.out_models_reg_plus_dir.parent,
            out_metrics_dir=cfg.out_metrics_dir,
            config=cfg,
        )

    _copy_summary_to_table_dir(
        [sign_metrics_dst, reg_metrics_dst, e2e_path, combined_path],
        cfg.table_summary_dir,
    )

    print(f"[DONE] Track pipeline complete: {args.track}")
    print(f"[DONE] Base output: {cfg.out_base_dir}")
    print(f"[DONE] Summary tables: {cfg.table_summary_dir}")


if __name__ == "__main__":
    main()
