#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from triplet_sampler import TripletSampler
from utils import (
    TRACK_DATASETS,
    default_phase5_index_path,
    ensure_reg_output_tree,
    group_split,
    load_XB_tensor,
    load_phase5_index,
    per_bin_regression_metrics,
    regression_metrics,
    resolve_reg_out_dir,
    save_json,
    split_train_val_by_group,
)


def _load_trusted_keras_model(path: str | Path) -> tf.keras.Model:
    # Pipeline checkpoints are generated locally and may include Lambda layers.
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)


@dataclass
class TrainArgs:
    track: str
    phase5_index: str
    out_dir: str | None
    seed: int
    split: tuple[float, float, float]
    batch_size: int
    epochs: int
    lr: float
    embedding_dim: int
    lambda2: float
    margin: float
    bin_width_um: float
    top_k_triplets_per_batch: int | None
    exclude_zero_from_triplets: bool
    mixed_precision: bool
    lodoo: bool
    resume: bool


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _inv_res_block(x, out_ch: int, stride: int, expand: int):
    in_ch = int(x.shape[-1])
    hidden = max(in_ch * int(expand), 8)

    y = x
    if expand != 1:
        y = tf.keras.layers.Conv2D(hidden, 1, padding="same", use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.ReLU(max_value=6.0)(y)

    y = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU(max_value=6.0)(y)

    y = tf.keras.layers.Conv2D(out_ch, 1, padding="same", use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)

    if stride == 1 and in_ch == out_ch:
        y = tf.keras.layers.Add()([x, y])
    return y


def build_regressor_model(embedding_dim: int = 128) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(200, 200, 4), name="XB")

    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    x = _inv_res_block(x, out_ch=16, stride=1, expand=1)
    x = _inv_res_block(x, out_ch=24, stride=2, expand=4)
    x = _inv_res_block(x, out_ch=24, stride=1, expand=3)
    x = _inv_res_block(x, out_ch=40, stride=2, expand=3)
    x = _inv_res_block(x, out_ch=40, stride=1, expand=3)
    x = _inv_res_block(x, out_ch=64, stride=2, expand=3)
    x = _inv_res_block(x, out_ch=64, stride=1, expand=3)

    x = tf.keras.layers.Conv2D(96, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    emb_raw = tf.keras.layers.Dense(int(embedding_dim), activation=None, name="embedding_dense")(gap)
    emb = tf.keras.layers.UnitNormalization(axis=-1, name="embedding_l2")(emb_raw)

    y_hat = tf.keras.layers.Dense(1, activation="softplus", dtype="float32", name="y_mag_hat_um")(gap)

    return tf.keras.Model(inputs=inp, outputs=[y_hat, emb], name="phase5_branch_regressor")


def triplet_loss(emb_a: tf.Tensor, emb_p: tf.Tensor, emb_n: tf.Tensor, margin: float) -> tf.Tensor:
    d_ap = tf.reduce_sum(tf.square(emb_a - emb_p), axis=-1)
    d_an = tf.reduce_sum(tf.square(emb_a - emb_n), axis=-1)
    loss = tf.nn.relu(d_ap - d_an + float(margin))
    return tf.reduce_mean(loss)


def load_xb_batch(df: pd.DataFrame, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch = df.iloc[indices]
    paths = batch["cache_path_XB"].astype(str).tolist()
    x = np.empty((len(paths), 200, 200, 4), dtype=np.float32)
    for i, p in enumerate(paths):
        x[i] = load_XB_tensor(p)
    y = batch["y_mag_um"].to_numpy(dtype=np.float32)
    return x, y


def predict_mags(model: tf.keras.Model, df: pd.DataFrame, batch_size: int) -> np.ndarray:
    preds: list[np.ndarray] = []
    n = len(df)
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n), dtype=np.int64)
        x, _ = load_xb_batch(df, idx)
        y_hat, _ = model(x, training=False)
        preds.append(tf.squeeze(y_hat, axis=-1).numpy())
    if not preds:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(preds, axis=0).astype(np.float32)


def evaluate_branch(model: tf.keras.Model, df: pd.DataFrame, bin_width_um: float) -> tuple[dict, pd.DataFrame]:
    y_true = df["y_mag_um"].to_numpy(dtype=np.float32)
    y_pred = predict_mags(model, df, batch_size=128)

    m = regression_metrics(y_true, y_pred)
    near_mask = y_true <= 2.0
    near_mae = float(np.mean(np.abs(y_true[near_mask] - y_pred[near_mask]))) if near_mask.any() else np.nan

    out = {
        "n_samples": int(len(df)),
        "mae_um": float(m["mae_um"]),
        "rmse_um": float(m["rmse_um"]),
        "near_focus_mae_um_y_le_2": near_mae,
    }
    per_bin = per_bin_regression_metrics(y_true=y_true, y_pred=y_pred, bin_width_um=bin_width_um)
    return out, per_bin


def train_one_branch(
    branch_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_paths: dict[str, Path],
    args: TrainArgs,
) -> tuple[Path, Path, Path, dict]:
    model = build_regressor_model(embedding_dim=args.embedding_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(args.lr))
    huber = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    sampler = TripletSampler(
        train_df,
        bin_width_um=float(args.bin_width_um),
        exclude_zero_from_triplets=bool(args.exclude_zero_from_triplets),
        seed=int(args.seed),
    )

    best_val_mae = np.inf
    wait_es = 0
    wait_lr = 0
    min_lr = 1e-6

    history_rows: list[dict] = []

    model_path = out_paths["models"] / f"R_{branch_name}_best.keras"
    hist_path = out_paths["metrics"] / f"history_{branch_name}.csv"
    eval_path = out_paths["metrics"] / f"eval_{branch_name}.json"
    per_bin_path = out_paths["metrics"] / f"per_bin_{branch_name}.csv"
    resume_state_path = out_paths["metrics"] / f"resume_state_{branch_name}.json"
    checkpoint_dir = out_paths["models"] / f"checkpoints_{branch_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(checkpoint_dir),
        max_to_keep=3,
    )

    start_epoch = 1
    if args.resume and checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        if hist_path.is_file():
            hist_df = pd.read_csv(hist_path)
            history_rows = hist_df.to_dict(orient="records")
        if resume_state_path.is_file():
            with resume_state_path.open("r", encoding="utf-8") as f:
                resume_state = json.load(f)
            best_val_mae = float(resume_state.get("best_val_mae", best_val_mae))
            wait_es = int(resume_state.get("wait_es", wait_es))
            wait_lr = int(resume_state.get("wait_lr", wait_lr))
            start_epoch = int(resume_state.get("last_completed_epoch", 0)) + 1
        elif history_rows:
            start_epoch = int(history_rows[-1]["epoch"]) + 1
        print(
            f"[INFO] Resume: restored branch={branch_name} from {checkpoint_manager.latest_checkpoint} "
            f"starting at epoch={start_epoch}."
        )

    def persist_epoch_state(last_completed_epoch: int) -> None:
        pd.DataFrame(history_rows).to_csv(hist_path, index=False)
        save_json(
            {
                "branch": branch_name,
                "last_completed_epoch": int(last_completed_epoch),
                "best_val_mae": float(best_val_mae),
                "wait_es": int(wait_es),
                "wait_lr": int(wait_lr),
                "lr": float(tf.keras.backend.get_value(optimizer.learning_rate)),
                "best_model_path": str(model_path),
                "latest_checkpoint": str(checkpoint_manager.save(checkpoint_number=int(last_completed_epoch))),
            },
            resume_state_path,
        )

    if start_epoch > int(args.epochs):
        print(
            f"[INFO] Resume: branch={branch_name} already completed through epoch {start_epoch - 1}; "
            f"skipping further training."
        )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        t0 = time.perf_counter()

        idx_all = np.arange(len(train_df), dtype=np.int64)
        np.random.default_rng(args.seed + epoch).shuffle(idx_all)

        reg_losses = []
        tri_losses = []
        total_losses = []
        triplets_epoch = 0

        for i in range(0, len(idx_all), int(args.batch_size)):
            b_idx = idx_all[i : i + int(args.batch_size)]
            if b_idx.size == 0:
                continue

            x_a_np, y_np = load_xb_batch(train_df, b_idx)
            y_tf = tf.convert_to_tensor(y_np, dtype=tf.float32)

            trips = sampler.sample_for_anchors(
                b_idx,
                top_k_triplets_per_batch=args.top_k_triplets_per_batch,
            )

            with tf.GradientTape() as tape:
                y_hat, emb_a_all = model(x_a_np, training=True)
                y_hat = tf.squeeze(tf.cast(y_hat, tf.float32), axis=-1)
                reg_loss = tf.reduce_mean(huber(y_tf, y_hat))

                tri_loss = tf.constant(0.0, dtype=tf.float32)
                if trips.shape[0] > 0:
                    triplets_epoch += int(trips.shape[0])
                    anchor_ids = trips[:, 0]
                    pos_ids = trips[:, 1]
                    neg_ids = trips[:, 2]

                    pos_in_batch = {int(g): j for j, g in enumerate(b_idx.tolist())}
                    a_pos = np.array([pos_in_batch[int(g)] for g in anchor_ids], dtype=np.int32)
                    emb_a = tf.gather(emb_a_all, a_pos)

                    x_p_np, _ = load_xb_batch(train_df, pos_ids)
                    x_n_np, _ = load_xb_batch(train_df, neg_ids)
                    _, emb_p = model(x_p_np, training=True)
                    _, emb_n = model(x_n_np, training=True)
                    tri_loss = triplet_loss(emb_a, emb_p, emb_n, margin=float(args.margin))

                total_loss = tf.cast(reg_loss, tf.float32) + float(args.lambda2) * tf.cast(tri_loss, tf.float32)

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            reg_losses.append(float(reg_loss.numpy()))
            tri_losses.append(float(tri_loss.numpy()))
            total_losses.append(float(total_loss.numpy()))

        val_eval, _ = evaluate_branch(model, val_df, bin_width_um=float(args.bin_width_um))
        val_mae = float(val_eval["mae_um"])
        val_rmse = float(val_eval["rmse_um"])

        lr_now = float(tf.keras.backend.get_value(optimizer.learning_rate))
        improved = val_mae < (best_val_mae - 1e-8)

        if improved:
            best_val_mae = val_mae
            wait_es = 0
            wait_lr = 0
            model.save(model_path)
        else:
            wait_es += 1
            wait_lr += 1

            if wait_lr >= 4:
                new_lr = max(min_lr, lr_now * 0.5)
                optimizer.learning_rate.assign(new_lr)
                wait_lr = 0

            if wait_es >= 10:
                history_rows.append(
                    {
                        "epoch": int(epoch),
                        "train_reg_loss": float(np.mean(reg_losses)) if reg_losses else np.nan,
                        "train_triplet_loss": float(np.mean(tri_losses)) if tri_losses else np.nan,
                        "train_total_loss": float(np.mean(total_losses)) if total_losses else np.nan,
                        "val_mae_um": val_mae,
                        "val_rmse_um": val_rmse,
                        "lr": float(tf.keras.backend.get_value(optimizer.learning_rate)),
                        "triplets_built": int(triplets_epoch),
                        "epoch_time_s": float(time.perf_counter() - t0),
                    }
                )
                persist_epoch_state(last_completed_epoch=epoch)
                print(f"[INFO] Early stopping branch={branch_name} at epoch={epoch}")
                break

        history_rows.append(
            {
                "epoch": int(epoch),
                "train_reg_loss": float(np.mean(reg_losses)) if reg_losses else np.nan,
                "train_triplet_loss": float(np.mean(tri_losses)) if tri_losses else np.nan,
                "train_total_loss": float(np.mean(total_losses)) if total_losses else np.nan,
                "val_mae_um": val_mae,
                "val_rmse_um": val_rmse,
                "lr": float(tf.keras.backend.get_value(optimizer.learning_rate)),
                "triplets_built": int(triplets_epoch),
                "epoch_time_s": float(time.perf_counter() - t0),
            }
        )
        persist_epoch_state(last_completed_epoch=epoch)

        print(
            f"[INFO] {branch_name} epoch={epoch} "
            f"train_reg={history_rows[-1]['train_reg_loss']:.5f} "
            f"train_tri={history_rows[-1]['train_triplet_loss']:.5f} "
            f"val_mae={val_mae:.5f} val_rmse={val_rmse:.5f}"
        )

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(hist_path, index=False)

    if model_path.is_file():
        best_model = _load_trusted_keras_model(model_path)
    else:
        best_model = model
        best_model.save(model_path)

    test_eval, test_bin = evaluate_branch(best_model, test_df, bin_width_um=float(args.bin_width_um))
    test_eval.update(
        {
            "branch": branch_name,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        }
    )
    save_json(test_eval, eval_path)
    test_bin.to_csv(per_bin_path, index=False)

    sampler_stats = sampler.stats.to_dict()
    sampler_stats.update({"branch": branch_name})

    return model_path, hist_path, eval_path, sampler_stats


def run_lodoo(df: pd.DataFrame, args: TrainArgs, out_paths: dict[str, Path]) -> None:
    results: list[dict] = []
    datasets = TRACK_DATASETS[args.track]
    val_ratio = args.split[1] / (args.split[0] + args.split[1] + 1e-12)

    lodoo_dir = out_paths["metrics"] / "lodoo_tmp"
    lodoo_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, held in enumerate(datasets):
        fold_test = df[df["dataset"].astype(str) == held].copy().reset_index(drop=True)
        fold_trainval = df[df["dataset"].astype(str) != held].copy().reset_index(drop=True)
        if fold_test.empty or fold_trainval.empty:
            continue

        for branch_name in ["plus", "minus"]:
            if branch_name == "plus":
                trv_b = fold_trainval[fold_trainval["defocus_um"].astype(float) > 0].copy().reset_index(drop=True)
                tst_b = fold_test[fold_test["defocus_um"].astype(float) > 0].copy().reset_index(drop=True)
            else:
                trv_b = fold_trainval[fold_trainval["defocus_um"].astype(float) < 0].copy().reset_index(drop=True)
                tst_b = fold_test[fold_test["defocus_um"].astype(float) < 0].copy().reset_index(drop=True)
            if trv_b.empty or tst_b.empty:
                continue

            try:
                tr_b, va_b = split_train_val_by_group(
                    trv_b,
                    val_ratio=float(val_ratio),
                    seed=args.seed + 1000 + fold_idx,
                )
            except Exception:
                continue

            local_paths = {
                "models": lodoo_dir,
                "metrics": lodoo_dir,
            }
            m_path, _, _, _ = train_one_branch(
                branch_name=f"{branch_name}_{held}",
                train_df=tr_b,
                val_df=va_b,
                test_df=tst_b,
                out_paths=local_paths,
                args=TrainArgs(**{**args.__dict__, "epochs": max(5, min(args.epochs, 20))}),
            )

            model = _load_trusted_keras_model(m_path)
            eval_dict, _ = evaluate_branch(model, tst_b, bin_width_um=float(args.bin_width_um))
            eval_dict.update({"held_out_dataset": held, "branch": branch_name})
            results.append(eval_dict)

    if results:
        out_csv = out_paths["metrics"] / "lodoo_regression_results.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"[DONE] Wrote LODOO regression results: {out_csv}")


def parse_args() -> TrainArgs:
    ap = argparse.ArgumentParser(description="Train Phase-5 branch regressors (R_plus, R_minus) with Huber + Triplet.")
    ap.add_argument("--track", required=True, choices=["smears", "biopsy"])
    ap.add_argument("--phase5-index", default=None, help="Path to index_phase5.csv")
    ap.add_argument("--out-dir", default=None, help="Output dir (must be under out_final_phase/<track>/regression)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--lambda2", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--bin-width-um", type=float, default=0.5)
    ap.add_argument("--top-k-triplets-per-batch", type=int, default=None)
    ap.add_argument("--exclude-zero-from-triplets", action="store_true", help="Exclude zero-defocus samples from triplet sampling")
    ap.add_argument("--include-zero-in-triplets", action="store_true", help="Override and include zero-defocus samples in triplets")
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--lodoo", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Reuse existing outputs and skip completed branch training.")
    a = ap.parse_args()

    exclude_zero = True
    if a.include_zero_in_triplets:
        exclude_zero = False
    elif a.exclude_zero_from_triplets:
        exclude_zero = True

    return TrainArgs(
        track=a.track,
        phase5_index=str(default_phase5_index_path(a.track) if a.phase5_index is None else a.phase5_index),
        out_dir=a.out_dir,
        seed=int(a.seed),
        split=(float(a.split[0]), float(a.split[1]), float(a.split[2])),
        batch_size=int(a.batch_size),
        epochs=int(a.epochs),
        lr=float(a.lr),
        embedding_dim=int(a.embedding_dim),
        lambda2=float(a.lambda2),
        margin=float(a.margin),
        bin_width_um=float(a.bin_width_um),
        top_k_triplets_per_batch=a.top_k_triplets_per_batch,
        exclude_zero_from_triplets=bool(exclude_zero),
        mixed_precision=bool(a.mixed_precision),
        lodoo=bool(a.lodoo),
        resume=bool(a.resume),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled")

    out_dir = resolve_reg_out_dir(args.track, args.out_dir)
    out_paths = ensure_reg_output_tree(out_dir)
    resume_targets = [
        out_paths["models"] / "R_plus_best.keras",
        out_paths["models"] / "R_minus_best.keras",
        out_paths["metrics"] / "history_plus.csv",
        out_paths["metrics"] / "history_minus.csv",
        out_paths["metrics"] / "eval_plus.json",
        out_paths["metrics"] / "eval_minus.json",
        out_paths["metrics"] / "per_bin_plus.csv",
        out_paths["metrics"] / "per_bin_minus.csv",
        out_paths["splits"] / "train_plus.csv",
        out_paths["splits"] / "val_plus.csv",
        out_paths["splits"] / "test_plus.csv",
        out_paths["splits"] / "train_minus.csv",
        out_paths["splits"] / "val_minus.csv",
        out_paths["splits"] / "test_minus.csv",
        out_paths["triplets"] / "triplet_config.json",
    ]
    if args.lodoo:
        resume_targets.append(out_paths["metrics"] / "lodoo_regression_results.csv")
    if args.resume and all(p.is_file() for p in resume_targets):
        print(f"[INFO] Resume: all Phase-5 training outputs already exist in {out_dir}; skipping.")
        return

    df = load_phase5_index(track=args.track, phase5_index_path=args.phase5_index)

    # Group-safe split over all rows first, then branch-filter.
    train_all, val_all, test_all = group_split(df, split=args.split, seed=args.seed)

    # branch: plus (>0), minus (<0)
    branches = ["plus", "minus"]
    trained_branch_models: dict[str, Path] = {}

    sampling_stats_rows = []

    for branch_name in branches:
        if branch_name == "plus":
            tr = train_all[train_all["defocus_um"].astype(float) > 0].copy().reset_index(drop=True)
            va = val_all[val_all["defocus_um"].astype(float) > 0].copy().reset_index(drop=True)
            te = test_all[test_all["defocus_um"].astype(float) > 0].copy().reset_index(drop=True)
        else:
            tr = train_all[train_all["defocus_um"].astype(float) < 0].copy().reset_index(drop=True)
            va = val_all[val_all["defocus_um"].astype(float) < 0].copy().reset_index(drop=True)
            te = test_all[test_all["defocus_um"].astype(float) < 0].copy().reset_index(drop=True)

        if tr.empty or va.empty or te.empty:
            print(
                f"[WARN] Branch '{branch_name}' has empty split subset: "
                f"train={len(tr)} val={len(va)} test={len(te)}. "
                f"Skipping direct training for this branch."
            )
            # Write empty split files for traceability.
            split_cols = [
                "roi_uid",
                "dataset",
                "group_id",
                "fov_id",
                "cache_path_XB",
                "roi_importance",
                "defocus_um",
                "y_sign",
                "y_mag_um",
            ]
            pd.DataFrame(columns=split_cols).to_csv(out_paths["splits"] / f"train_{branch_name}.csv", index=False)
            pd.DataFrame(columns=split_cols).to_csv(out_paths["splits"] / f"val_{branch_name}.csv", index=False)
            pd.DataFrame(columns=split_cols).to_csv(out_paths["splits"] / f"test_{branch_name}.csv", index=False)
            continue

        split_cols = [
            "roi_uid",
            "dataset",
            "group_id",
            "fov_id",
            "cache_path_XB",
            "roi_importance",
            "defocus_um",
            "y_sign",
            "y_mag_um",
        ]
        split_train_path = out_paths["splits"] / f"train_{branch_name}.csv"
        split_val_path = out_paths["splits"] / f"val_{branch_name}.csv"
        split_test_path = out_paths["splits"] / f"test_{branch_name}.csv"

        if args.resume and split_train_path.is_file() and split_val_path.is_file() and split_test_path.is_file():
            print(f"[INFO] Resume: split files already exist for branch={branch_name}; keeping.")
        else:
            tr[split_cols].to_csv(split_train_path, index=False)
            va[split_cols].to_csv(split_val_path, index=False)
            te[split_cols].to_csv(split_test_path, index=False)

        branch_outputs = [
            out_paths["models"] / f"R_{branch_name}_best.keras",
            out_paths["metrics"] / f"history_{branch_name}.csv",
            out_paths["metrics"] / f"eval_{branch_name}.json",
            out_paths["metrics"] / f"per_bin_{branch_name}.csv",
        ]
        if args.resume and all(p.is_file() for p in branch_outputs):
            print(f"[INFO] Resume: branch outputs exist for {branch_name}; skipping training.")
            trained_branch_models[branch_name] = out_paths["models"] / f"R_{branch_name}_best.keras"
            continue

        m_path, h_path, e_path, s_stats = train_one_branch(
            branch_name=branch_name,
            train_df=tr,
            val_df=va,
            test_df=te,
            out_paths=out_paths,
            args=args,
        )
        print(f"[DONE] Branch={branch_name} model: {m_path}")
        print(f"[DONE] Branch={branch_name} history: {h_path}")
        print(f"[DONE] Branch={branch_name} eval: {e_path}")
        sampling_stats_rows.append(s_stats)
        trained_branch_models[branch_name] = Path(m_path)

    # Handle single-sign tracks: if one branch is missing, clone trained model and
    # create placeholder metrics/history so downstream inference can proceed.
    if not trained_branch_models:
        raise ValueError("No branch could be trained. Check data coverage and split settings.")

    for branch_name in branches:
        if branch_name in trained_branch_models:
            continue
        donor = "plus" if "plus" in trained_branch_models else "minus"
        donor_model = trained_branch_models[donor]
        target_model = out_paths["models"] / f"R_{branch_name}_best.keras"
        shutil.copy2(donor_model, target_model)
        print(f"[WARN] Branch '{branch_name}' missing. Cloned model from '{donor}' -> {target_model}")

        # Placeholder training artifacts
        hist_path = out_paths["metrics"] / f"history_{branch_name}.csv"
        if not hist_path.is_file():
            pd.DataFrame(
                [{"epoch": 0, "train_reg_loss": np.nan, "train_triplet_loss": np.nan, "train_total_loss": np.nan, "val_mae_um": np.nan, "val_rmse_um": np.nan, "lr": np.nan, "triplets_built": 0, "epoch_time_s": 0.0}]
            ).to_csv(hist_path, index=False)

        eval_path = out_paths["metrics"] / f"eval_{branch_name}.json"
        if not eval_path.is_file():
            save_json(
                {
                    "branch": branch_name,
                    "note": f"fallback_cloned_from_{donor}",
                    "n_samples": 0,
                    "mae_um": np.nan,
                    "rmse_um": np.nan,
                    "near_focus_mae_um_y_le_2": np.nan,
                },
                eval_path,
            )

        per_bin_path = out_paths["metrics"] / f"per_bin_{branch_name}.csv"
        if not per_bin_path.is_file():
            pd.DataFrame(columns=["bin_lo_um", "bin_hi_um", "count", "mae_um", "rmse_um"]).to_csv(per_bin_path, index=False)

    triplet_cfg = {
        "track": args.track,
        "seed": args.seed,
        "split": list(args.split),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "embedding_dim": args.embedding_dim,
        "lambda1": 1.0,
        "lambda2": args.lambda2,
        "margin": args.margin,
        "bin_width_um": args.bin_width_um,
        "exclude_zero_from_triplets": args.exclude_zero_from_triplets,
        "top_k_triplets_per_batch": args.top_k_triplets_per_batch,
    }
    triplet_cfg_path = out_paths["triplets"] / "triplet_config.json"
    if args.resume and triplet_cfg_path.is_file():
        print(f"[INFO] Resume: triplet config already exists at {triplet_cfg_path}; keeping.")
    else:
        save_json(triplet_cfg, triplet_cfg_path)

    if sampling_stats_rows:
        pd.DataFrame(sampling_stats_rows).to_csv(out_paths["triplets"] / "sampling_stats.csv", index=False)

    if args.lodoo:
        run_lodoo(df=df, args=args, out_paths=out_paths)


if __name__ == "__main__":
    main()
