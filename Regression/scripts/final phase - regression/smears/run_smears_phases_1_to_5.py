#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils_smears import command_runner, safe_mkdir

ROOT = Path("/home/dineth/focus_measure/journal/Regression")
SCRIPTS_ROOT = ROOT / "scripts" / "final phase - regression"
SMEARS_SCRIPTS = SCRIPTS_ROOT / "smears"
DATA_DIR = ROOT / "data"
OUT_SMEARS = DATA_DIR / "out_final_phase" / "smears"
TABLES_SMEARS = ROOT / "tables" / "FINAL_PHASE" / "SMEARS"

MANIFEST_PBS = DATA_DIR / "manifest_pbs.csv"
MANIFEST_WBC = DATA_DIR / "manifest_wbc.csv"
MANIFEST_BMA = DATA_DIR / "manifest_bma.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run SMEARS Phases 1-5 end-to-end orchestration.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument("--top-k", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs-sign", type=int, default=80)
    ap.add_argument("--epochs-reg", type=int, default=120)
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume-safe mode; pass --resume to all phase scripts.")
    return ap.parse_args()


def _resolve_phase3_script() -> Path:
    cands = [
        SCRIPTS_ROOT / "phase3_preprocess" / "phase3_build_cache.py",
        SCRIPTS_ROOT / "phase3_preprocessing" / "phase3_build_cache.py",
    ]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "Missing Phase 3 cache builder script. Expected one of:\n"
        + "\n".join(str(c) for c in cands)
    )


def _validate_inputs_and_scripts(phase3_script: Path) -> None:
    missing_inputs = [str(p) for p in [MANIFEST_PBS, MANIFEST_WBC, MANIFEST_BMA] if not p.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing required dataset manifests:\n" + "\n".join(missing_inputs)
        )

    required_scripts = [
        SMEARS_SCRIPTS / "build_manifest_smears_all.py",
        SMEARS_SCRIPTS / "make_smears_splits.py",
        phase3_script,
        SCRIPTS_ROOT / "phase4_sign" / "train_sign.py",
        SCRIPTS_ROOT / "phase4_sign" / "calibrate_tau.py",
        SCRIPTS_ROOT / "phase4_sign" / "infer_vote_sign.py",
        SCRIPTS_ROOT / "phase5_regression" / "build_phase5_index.py",
        SCRIPTS_ROOT / "phase5_regression" / "train_regressors.py",
        SCRIPTS_ROOT / "phase5_regression" / "infer_regress_and_aggregate.py",
    ]
    missing_scripts = [str(p) for p in required_scripts if not p.is_file()]
    if missing_scripts:
        raise FileNotFoundError("Missing required phase scripts:\n" + "\n".join(missing_scripts))


def main() -> None:
    args = parse_args()

    split = [float(args.split[0]), float(args.split[1]), float(args.split[2])]
    if abs(sum(split) - 1.0) > 1e-8:
        raise ValueError(f"--split must sum to 1.0, got {split}")

    safe_mkdir(OUT_SMEARS)
    safe_mkdir(TABLES_SMEARS)
    log_path = OUT_SMEARS / "run_log.txt"

    phase3_script = _resolve_phase3_script()
    _validate_inputs_and_scripts(phase3_script)

    unified_manifest = DATA_DIR / "manifest_smears_all.csv"
    splits_dir = OUT_SMEARS / "splits"
    cache_index = DATA_DIR / "cache_phase3" / "smears" / "cache_index.csv"
    sign_out_dir = OUT_SMEARS / "sign"
    reg_out_dir = OUT_SMEARS / "regression"
    phase5_index = reg_out_dir / "index_phase5.csv"

    py = sys.executable

    cmds: list[list[str]] = [
        [py, str(SMEARS_SCRIPTS / "build_manifest_smears_all.py")],
        [
            py,
            str(SMEARS_SCRIPTS / "make_smears_splits.py"),
            "--manifest",
            str(unified_manifest),
            "--out-dir",
            str(splits_dir),
            "--seed",
            str(args.seed),
            "--split",
            str(split[0]),
            str(split[1]),
            str(split[2]),
        ],
        [
            py,
            str(phase3_script),
            "--track",
            "smears",
            "--datasets",
            "pbs",
            "wbc",
            "bma",
            "--manifest-paths",
            str(MANIFEST_PBS),
            str(MANIFEST_WBC),
            str(MANIFEST_BMA),
        ],
        [
            py,
            str(SCRIPTS_ROOT / "phase4_sign" / "train_sign.py"),
            "--track",
            "smears",
            "--cache-index",
            str(cache_index),
            "--seed",
            str(args.seed),
            "--split",
            str(split[0]),
            str(split[1]),
            str(split[2]),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs_sign),
        ],
        [
            py,
            str(SCRIPTS_ROOT / "phase4_sign" / "calibrate_tau.py"),
            "--track",
            "smears",
            "--cache-index",
            str(cache_index),
            "--splits-dir",
            str(sign_out_dir / "splits"),
        ],
        [
            py,
            str(SCRIPTS_ROOT / "phase4_sign" / "infer_vote_sign.py"),
            "--track",
            "smears",
            "--cache-index",
            str(cache_index),
            "--mode",
            "auto",
            "--top-k",
            str(args.top_k),
        ],
        [
            py,
            str(SCRIPTS_ROOT / "phase5_regression" / "build_phase5_index.py"),
            "--track",
            "smears",
            "--cache-index",
            str(cache_index),
        ],
        [
            py,
            str(SCRIPTS_ROOT / "phase5_regression" / "train_regressors.py"),
            "--track",
            "smears",
            "--phase5-index",
            str(phase5_index),
            "--seed",
            str(args.seed),
            "--split",
            str(split[0]),
            str(split[1]),
            str(split[2]),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs_reg),
        ],
        [
            py,
            str(SCRIPTS_ROOT / "phase5_regression" / "infer_regress_and_aggregate.py"),
            "--track",
            "smears",
            "--phase5-index",
            str(phase5_index),
            "--sign-out-dir",
            str(sign_out_dir),
            "--reg-out-dir",
            str(reg_out_dir),
            "--top-k",
            str(args.top_k),
        ],
    ]

    if args.mixed_precision:
        # apply to train scripts only
        cmds[3].append("--mixed-precision")
        cmds[7].append("--mixed-precision")

    if args.resume:
        # Apply resume semantics to every phase command that supports it.
        for idx in range(len(cmds)):
            cmds[idx].append("--resume")

    print("[INFO] SMEARS pipeline commands (Phases 1-5):")
    for c in cmds:
        print(" ", " ".join(c))

    # clear previous log for deterministic run logs
    if log_path.exists():
        log_path.unlink()

    for idx, cmd in enumerate(cmds, start=1):
        command_runner(cmd, log_path=log_path)

        # hard checkpoints
        if idx == 1 and not unified_manifest.is_file():
            raise FileNotFoundError(f"Unified manifest was not created: {unified_manifest}")
        if idx == 2 and not (splits_dir / "split_config.json").is_file() and not (splits_dir / "train.csv").is_file():
            raise FileNotFoundError(f"SMEARS splits were not created in: {splits_dir}")
        if idx == 3 and not cache_index.is_file():
            raise FileNotFoundError(
                f"Phase 3 did not produce cache index: {cache_index}. "
                f"Run Phase 3 to generate cache before Phase 4."
            )

    print(f"[DONE] SMEARS run complete. Log: {log_path}")


if __name__ == "__main__":
    main()
