#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf

from utils import (
    TRACK_DATASETS,
    build_sign_labels,
    compute_balanced_accuracy,
    compute_per_bin_metrics,
    default_cache_index_path,
    ensure_output_tree,
    group_split,
    load_XA_tensor,
    load_cache_index,
    resolve_out_dir,
    save_json,
    split_train_val_by_group,
)


@dataclass
class TrainArgs:
    track: str
    cache_index: str
    out_dir: str | None
    seed: int
    split: tuple[float, float, float]
    batch_size: int
    epochs: int
    lr: float
    dropout: float
    lodoo: bool
    max_samples: int | None
    num_workers: int
    mixed_precision: bool
    resume: bool


class XASequence(tf.keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame, batch_size: int, shuffle: bool, seed: int):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.df), dtype=np.int64)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx: int):
        i0 = idx * self.batch_size
        i1 = min((idx + 1) * self.batch_size, len(self.df))
        batch_idx = self.indices[i0:i1]
        batch = self.df.iloc[batch_idx]

        x = np.empty((len(batch), 200, 200, 3), dtype=np.float32)
        y = batch["y_sign"].to_numpy(dtype=np.float32)
        for i, p in enumerate(batch["cache_path_XA"].astype(str).tolist()):
            x[i] = load_XA_tensor(p)
        return x, y

    def on_epoch_end(self) -> None:
        if self.shuffle and len(self.indices) > 1:
            self.rng.shuffle(self.indices)


def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(lr: float, dropout: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(200, 200, 3), name="XA")
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(200, 200, 3),
        include_top=False,
        weights=None,
        pooling="avg",
    )
    x = base(inp)
    x = tf.keras.layers.Dropout(float(dropout))(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="sign_prob")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def compute_class_weights(train_df: pd.DataFrame) -> dict[int, float]:
    vc = train_df["y_sign"].value_counts().to_dict()
    n0 = int(vc.get(0, 0))
    n1 = int(vc.get(1, 0))
    if n0 == 0 or n1 == 0:
        raise ValueError(f"Both classes are required in train split. Found counts: {vc}")

    total = n0 + n1
    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    return {0: float(w0), 1: float(w1)}


def predict_probs(model: tf.keras.Model, df: pd.DataFrame, batch_size: int) -> np.ndarray:
    probs: list[np.ndarray] = []
    paths = df["cache_path_XA"].astype(str).tolist()
    for i in range(0, len(paths), batch_size):
        sub = paths[i : i + batch_size]
        x = np.empty((len(sub), 200, 200, 3), dtype=np.float32)
        for j, p in enumerate(sub):
            x[j] = load_XA_tensor(p)
        p = model.predict(x, verbose=0).reshape(-1)
        probs.append(p)
    return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)


def evaluate_split(model: tf.keras.Model, df: pd.DataFrame, batch_size: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y_true = df["y_sign"].to_numpy(dtype=int)
    probs = predict_probs(model, df, batch_size=batch_size)
    y_pred = (probs >= 0.5).astype(int)

    auc = tf.keras.metrics.AUC()
    auc.update_state(y_true.astype(np.float32), probs.astype(np.float32))
    auroc = float(auc.result().numpy())

    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    bal_acc = float(compute_balanced_accuracy(y_true, y_pred)) if len(y_true) else 0.0

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))

    eval_dict = {
        "n_test": int(len(df)),
        "auroc": auroc,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )

    bins_df = compute_per_bin_metrics(
        y_true=y_true,
        y_pred=y_pred,
        abs_delta=df["abs_delta_z"].to_numpy(dtype=float),
    )

    return eval_dict, cm_df, bins_df


def fit_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_path: Path,
    history_path: Path,
    args: TrainArgs,
) -> tf.keras.Model:
    train_seq = XASequence(train_df, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_seq = XASequence(val_df, batch_size=args.batch_size, shuffle=False, seed=args.seed)

    model = build_model(lr=args.lr, dropout=args.dropout)
    class_weights = compute_class_weights(train_df)
    backup_dir = history_path.parent / f"backup_state_{model_path.stem}"
    csv_append = bool(args.resume and history_path.is_file())

    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(backup_dir),
            delete_checkpoint=False,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(history_path),
            separator=",",
            append=csv_append,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    fit_kwargs = {}
    if args.num_workers > 0:
        fit_kwargs["workers"] = int(args.num_workers)
        fit_kwargs["use_multiprocessing"] = bool(args.num_workers > 1)

    try:
        hist = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=int(args.epochs),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2,
            **fit_kwargs,
        )
    except TypeError:
        # Fallback for TF/Keras builds that do not accept workers/use_multiprocessing.
        hist = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=int(args.epochs),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2,
        )

    print(f"[DONE] Wrote history: {history_path}")

    if model_path.is_file():
        best = tf.keras.models.load_model(model_path)
        return best
    return model


def run_lodoo(df: pd.DataFrame, args: TrainArgs, out_paths: dict[str, Path]) -> None:
    results = []
    datasets = sorted(df["dataset"].astype(str).unique().tolist())
    val_ratio = args.split[1] / (args.split[0] + args.split[1] + 1e-12)

    for i, held_out in enumerate(datasets):
        fold_test = df[df["dataset"].astype(str) == held_out].copy().reset_index(drop=True)
        fold_trainval = df[df["dataset"].astype(str) != held_out].copy().reset_index(drop=True)
        if fold_test.empty or fold_trainval.empty:
            print(f"[WARN] Skipping LODOO fold={held_out} due to empty train/test")
            continue

        fold_train, fold_val = split_train_val_by_group(
            fold_trainval,
            val_ratio=float(val_ratio),
            seed=int(args.seed + i + 1000),
        )

        fold_dir = out_paths["metrics"] / "lodoo" / held_out
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_model_path = fold_dir / "best_model.keras"
        fold_hist_path = fold_dir / "history.csv"
        fold_args = TrainArgs(**{**args.__dict__, "seed": int(args.seed + i + 1000)})

        model = fit_model(
            train_df=fold_train,
            val_df=fold_val,
            model_path=fold_model_path,
            history_path=fold_hist_path,
            args=fold_args,
        )

        eval_dict, _, _ = evaluate_split(model, fold_test, batch_size=args.batch_size)
        eval_dict["held_out_dataset"] = held_out
        eval_dict["n_train"] = int(len(fold_train))
        eval_dict["n_val"] = int(len(fold_val))
        eval_dict["n_test"] = int(len(fold_test))
        results.append(eval_dict)

    if results:
        out_csv = out_paths["metrics"] / "lodoo_results.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"[DONE] Wrote LODOO results: {out_csv}")


def parse_args() -> TrainArgs:
    ap = argparse.ArgumentParser(description="Phase-4 sign training (Stage A: XA=[I,D1,D2]).")
    ap.add_argument("--track", required=True, choices=["smears", "biopsy"])
    ap.add_argument("--cache-index", default=None, help="Path to cache_index.csv")
    ap.add_argument("--out-dir", default=None, help="Output dir (must be within out_final_phase/<track>/sign)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lodoo", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Reuse existing outputs and skip completed steps.")
    a = ap.parse_args()

    split = tuple(float(x) for x in a.split)
    return TrainArgs(
        track=a.track,
        cache_index=str(default_cache_index_path(a.track) if a.cache_index is None else a.cache_index),
        out_dir=a.out_dir,
        seed=int(a.seed),
        split=split,
        batch_size=int(a.batch_size),
        epochs=int(a.epochs),
        lr=float(a.lr),
        dropout=float(a.dropout),
        lodoo=bool(a.lodoo),
        max_samples=a.max_samples,
        num_workers=int(a.num_workers),
        mixed_precision=bool(a.mixed_precision),
        resume=bool(a.resume),
    )


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled: mixed_float16")

    out_dir = resolve_out_dir(args.track, args.out_dir)
    out_paths = ensure_output_tree(out_dir)
    model_path = out_paths["models"] / "best_model.keras"
    history_path = out_paths["metrics"] / "history.csv"
    eval_path = out_paths["metrics"] / "eval.json"
    cm_path = out_paths["metrics"] / "confusion_matrix.csv"
    bins_path = out_paths["metrics"] / "per_bin_metrics.csv"
    split_paths = [out_paths["splits"] / "train.csv", out_paths["splits"] / "val.csv", out_paths["splits"] / "test.csv"]
    lodoo_path = out_paths["metrics"] / "lodoo_results.csv"

    if args.resume:
        resume_targets = [model_path, history_path, eval_path, cm_path, bins_path, *split_paths]
        if args.lodoo:
            resume_targets.append(lodoo_path)
        if all(p.is_file() for p in resume_targets):
            print(f"[INFO] Resume: all Stage-4 outputs already exist in {out_dir}; skipping.")
            return

    df = load_cache_index(track=args.track, cache_index_path=args.cache_index)
    df = build_sign_labels(df)

    allowed = set(TRACK_DATASETS[args.track])
    if not set(df["dataset"].astype(str).unique().tolist()).issubset(allowed):
        raise ValueError("Track/dataset mixing detected in cache index.")

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be > 0")
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed).reset_index(drop=True)
        print(f"[INFO] Using max-samples={len(df)}")

    train_df, val_df, test_df = group_split(df, split=args.split, seed=args.seed)

    split_cols = [
        "roi_uid",
        "dataset",
        "group_id",
        "cache_path_XA",
        "delta_z",
        "abs_delta_z",
        "y_sign",
        "roi_importance",
        "source_image_path",
    ]
    for sp_name, sp_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        cols = [c for c in split_cols if c in sp_df.columns]
        out_csv = out_paths["splits"] / f"{sp_name}.csv"
        if args.resume and out_csv.is_file():
            print(f"[INFO] Resume: split exists, keeping {out_csv}")
        else:
            sp_df[cols].to_csv(out_csv, index=False)
            print(f"[DONE] Wrote split: {out_csv} rows={len(sp_df)}")

    if args.resume and model_path.is_file() and history_path.is_file():
        backup_dir = history_path.parent / f"backup_state_{model_path.stem}"
        if not backup_dir.exists():
            print(f"[INFO] Resume: reusing existing sign model and history: {model_path}")
            model = tf.keras.models.load_model(model_path)
        else:
            print(f"[INFO] Resume: found in-progress training state at {backup_dir}; resuming fit().")
            model = fit_model(
                train_df=train_df,
                val_df=val_df,
                model_path=model_path,
                history_path=history_path,
                args=args,
            )
    else:
        model = fit_model(
            train_df=train_df,
            val_df=val_df,
            model_path=model_path,
            history_path=history_path,
            args=args,
        )

    eval_dict, cm_df, bins_df = evaluate_split(model, test_df, batch_size=args.batch_size)
    eval_dict.update(
        {
            "seed": args.seed,
            "split": list(args.split),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        }
    )

    save_json(eval_dict, eval_path)
    cm_df.to_csv(cm_path, index=True)
    bins_df.to_csv(bins_path, index=False)

    print(f"[DONE] Wrote eval: {eval_path}")
    print(f"[DONE] Wrote confusion matrix: {cm_path}")
    print(f"[DONE] Wrote per-bin metrics: {bins_path}")
    print(f"[DONE] Best model: {model_path}")

    if args.lodoo:
        run_lodoo(df=df, args=args, out_paths=out_paths)


if __name__ == "__main__":
    main()
