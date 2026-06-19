#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model_direct_regression import build_direct_regression_model
from utils_direct_regression import (
    ensure_direct_tree,
    evaluate_direct_predictions,
    load_or_create_direct_splits,
    load_xb_batch,
    per_bin_fov_metrics,
    predict_roi,
    save_eval_json,
    set_seed,
    write_split_files,
    aggregate_fov_predictions,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Train direct single-stage signed-distance regression baseline')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split', nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--top-k', type=int, default=7)
    ap.add_argument('--mixed-precision', action='store_true')
    ap.add_argument('--resume', action='store_true')
    return ap.parse_args()


def _persist_state(history_rows: list[dict], state_path: Path, best_val_fov_mae: float, wait_es: int, wait_lr: int, last_epoch: int) -> None:
    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(state_path.parent / 'train_history.csv', index=False)
    payload = {
        'best_val_fov_mae': float(best_val_fov_mae),
        'wait_es': int(wait_es),
        'wait_lr': int(wait_lr),
        'last_completed_epoch': int(last_epoch),
    }
    with state_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    out_paths = ensure_direct_tree(args.track)
    model_path = out_paths['models'] / 'best_model.keras'
    history_path = out_paths['metrics'] / 'train_history.csv'
    eval_path = out_paths['metrics'] / 'eval.json'
    per_bin_path = out_paths['metrics'] / 'per_bin.csv'
    state_path = out_paths['metrics'] / 'resume_state.json'

    final_outputs = [model_path, history_path, eval_path, per_bin_path]
    if args.resume and all(p.is_file() for p in final_outputs):
        print(f'[INFO] Resume: direct regression outputs already exist in {out_paths["root"]}; skipping.')
        return

    train_df, val_df, test_df, split_meta, index_df = load_or_create_direct_splits(
        args.track,
        seed=int(args.seed),
        split=(float(args.split[0]), float(args.split[1]), float(args.split[2])),
        resume=bool(args.resume),
    )
    write_split_files(out_paths, train_df, val_df, test_df)

    model = build_direct_regression_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(args.lr), clipnorm=1.0)
    loss_fn = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_dir = out_paths['models'] / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=str(ckpt_dir), max_to_keep=3)

    history_rows: list[dict] = []
    best_val_fov_mae = np.inf
    wait_es = 0
    wait_lr = 0
    min_lr = 1e-6
    start_epoch = 1

    if args.resume and ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        if history_path.is_file():
            history_rows = pd.read_csv(history_path).to_dict(orient='records')
        if state_path.is_file():
            with state_path.open('r', encoding='utf-8') as f:
                st = json.load(f)
            best_val_fov_mae = float(st.get('best_val_fov_mae', best_val_fov_mae))
            wait_es = int(st.get('wait_es', 0))
            wait_lr = int(st.get('wait_lr', 0))
            start_epoch = int(st.get('last_completed_epoch', 0)) + 1
        elif history_rows:
            start_epoch = int(history_rows[-1]['epoch']) + 1
        print(f'[INFO] Resume: restored direct regression training from {ckpt_manager.latest_checkpoint}, next_epoch={start_epoch}')

    for epoch in range(start_epoch, int(args.epochs) + 1):
        t0 = time.perf_counter()
        order = np.random.permutation(len(train_df))
        batch_losses: list[float] = []
        for i in range(0, len(order), int(args.batch_size)):
            idx = order[i:i + int(args.batch_size)]
            x_np, y_np = load_xb_batch(train_df, idx)
            y_tf = tf.convert_to_tensor(y_np, dtype=tf.float32)
            with tf.GradientTape() as tape:
                y_hat = tf.squeeze(tf.cast(model(x_np, training=True), tf.float32), axis=-1)
                loss = tf.reduce_mean(loss_fn(y_tf, y_hat))
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else None for g in grads]
            optimizer.apply_gradients([(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
            batch_losses.append(float(loss.numpy()))

        val_roi = val_df.copy().reset_index(drop=True)
        val_roi['y_pred_signed_um'] = predict_roi(model, val_roi, batch_size=128)
        val_fov = aggregate_fov_predictions(val_roi, top_k=int(args.top_k), default_method='weighted_mean')
        val_metrics = evaluate_direct_predictions(val_roi, val_fov)
        val_fov_mae = float(val_metrics['fov_mae_um'])
        lr_now = float(tf.keras.backend.get_value(optimizer.learning_rate))

        improved = np.isfinite(val_fov_mae) and (val_fov_mae < best_val_fov_mae - 1e-8)
        if improved:
            best_val_fov_mae = val_fov_mae
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
                        'epoch': int(epoch),
                        'train_loss': float(np.mean(batch_losses)) if batch_losses else np.nan,
                        'val_roi_mae_um': float(val_metrics['roi_mae_um']),
                        'val_fov_mae_um': float(val_metrics['fov_mae_um']),
                        'val_fov_rmse_um': float(val_metrics['fov_rmse_um']),
                        'lr': float(tf.keras.backend.get_value(optimizer.learning_rate)),
                        'epoch_time_s': float(time.perf_counter() - t0),
                    }
                )
                _persist_state(history_rows, state_path, best_val_fov_mae, wait_es, wait_lr, epoch)
                print(f'[INFO] Early stopping direct regression at epoch={epoch}')
                break

        history_rows.append(
            {
                'epoch': int(epoch),
                'train_loss': float(np.mean(batch_losses)) if batch_losses else np.nan,
                'val_roi_mae_um': float(val_metrics['roi_mae_um']),
                'val_fov_mae_um': float(val_metrics['fov_mae_um']),
                'val_fov_rmse_um': float(val_metrics['fov_rmse_um']),
                'lr': float(tf.keras.backend.get_value(optimizer.learning_rate)),
                'epoch_time_s': float(time.perf_counter() - t0),
            }
        )
        _persist_state(history_rows, state_path, best_val_fov_mae, wait_es, wait_lr, epoch)
        ckpt_manager.save(checkpoint_number=epoch)
        print(
            f'[INFO] epoch={epoch} train_loss={history_rows[-1]["train_loss"]:.5f} '
            f'val_fov_mae={history_rows[-1]["val_fov_mae_um"]:.5f} '
            f'val_fov_rmse={history_rows[-1]["val_fov_rmse_um"]:.5f}'
        )

    best_model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False) if model_path.is_file() else model
    test_roi = test_df.copy().reset_index(drop=True)
    test_roi['y_pred_signed_um'] = predict_roi(best_model, test_roi, batch_size=128)
    roi_runtime_start = time.perf_counter()
    _ = predict_roi(best_model, test_df.head(min(256, len(test_df))).copy(), batch_size=128) if len(test_df) > 0 else np.zeros((0,), dtype=np.float32)
    roi_runtime_ms = 1000.0 * (time.perf_counter() - roi_runtime_start) / max(min(256, len(test_df)), 1)

    test_fov = aggregate_fov_predictions(test_roi, top_k=int(args.top_k), default_method='weighted_mean')
    test_fov['runtime_ms_per_fov'] = float(roi_runtime_ms) * pd.to_numeric(test_fov['num_inputs_used'], errors='coerce').fillna(0.0)
    metrics = evaluate_direct_predictions(test_roi, test_fov)
    metrics.update(
        {
            'track': args.track,
            'top_k': int(args.top_k),
            'split_source': split_meta.get('source', 'unknown'),
            'n_train': int(len(train_df)),
            'n_val': int(len(val_df)),
            'n_test': int(len(test_df)),
            'runtime_ms_per_roi': float(roi_runtime_ms),
        }
    )
    save_eval_json(metrics, eval_path)
    per_bin_fov_metrics(test_fov, bin_width_um=0.5).to_csv(per_bin_path, index=False)
    print(f'[DONE] Direct regression model: {model_path}')
    print(f'[DONE] Direct regression eval: {eval_path}')


if __name__ == '__main__':
    main()
