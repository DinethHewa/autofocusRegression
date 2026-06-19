#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("TensorFlow is required for train_sign.py") from exc


def _gaussian_kernel(size: int, sigma: float) -> tf.Tensor:
    x = tf.range(size, dtype=tf.float32) - (size - 1) / 2.0
    g = tf.exp(-(x ** 2) / (2.0 * sigma ** 2))
    g = g / tf.reduce_sum(g)
    kernel_2d = tf.tensordot(g, g, axes=0)
    kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]
    return kernel_2d


def _gaussian_blur(gray: tf.Tensor, sigma: float) -> tf.Tensor:
    ksize = int(max(3, 2 * int(3 * sigma) + 1))
    kernel = _gaussian_kernel(ksize, sigma)
    return tf.nn.depthwise_conv2d(gray, tf.tile(kernel, [1, 1, 1, 1]), strides=[1, 1, 1, 1], padding="SAME")


def _build_stage_a_tensor(img: tf.Tensor, image_size: int, sigma1: float, sigma2: float) -> tf.Tensor:
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.image.convert_image_dtype(img, tf.float32)
    gray = tf.image.rgb_to_grayscale(img)

    blur1 = _gaussian_blur(gray[tf.newaxis, ...], sigma1)[0]
    blur2 = _gaussian_blur(gray[tf.newaxis, ...], sigma2)[0]

    d1 = gray - blur1
    d2 = blur1 - blur2

    i_ch = tf.clip_by_value(gray, 0.0, 1.0)
    d1_ch = tf.clip_by_value((d1 + 0.5), 0.0, 1.0)
    d2_ch = tf.clip_by_value((d2 + 0.5), 0.0, 1.0)
    return tf.concat([i_ch, d1_ch, d2_ch], axis=-1)


def _build_sign_model(image_size: int, lr: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="stage_a")
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=None,
        pooling="avg",
    )
    x = backbone(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="sign_prob")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auroc")],
    )
    return model


def _prepare_sign_df(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work[work["delta_z"] != 0].copy()
    if work.empty:
        raise ValueError("No non-zero delta_z rows available for sign training.")
    work["sign_label"] = (work["y_sign"] > 0).astype(np.float32)
    if "roi_importance" not in work.columns:
        work["roi_importance"] = 1.0
    work["roi_importance"] = pd.to_numeric(work["roi_importance"], errors="coerce").fillna(1.0)
    return work


def _to_dataset(df: pd.DataFrame, config, training: bool) -> tf.data.Dataset:
    paths = df["image_path"].astype(str).values
    labels = df["sign_label"].astype(np.float32).values
    weights = df["roi_importance"].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))

    def _map(path, label, weight):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        x = _build_stage_a_tensor(img, config.train.image_size, config.sigma1, config.sigma2)
        return x, label, weight

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(4096, len(df)), seed=config.split_seed, reshuffle_each_iteration=True)
    ds = ds.batch(config.train.sign_batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = float((y_pred[pos_mask] == 1).mean()) if pos_mask.any() else 0.0
    tnr = float((y_pred[neg_mask] == 0).mean()) if neg_mask.any() else 0.0
    return 0.5 * (tpr + tnr)


def _per_bin_sign_accuracy(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    bins = [0.0, 0.5, 1.0, 2.0, np.inf]
    labels = ["(0,0.5]", "(0.5,1.0]", "(1.0,2.0]", "(2.0,inf]"]
    out = []
    mags = df["y_mag"].to_numpy()
    y_true = (df["y_sign"].to_numpy() > 0).astype(int)
    for lo, hi, name in zip(bins[:-1], bins[1:], labels):
        idx = (mags > lo) & (mags <= hi)
        if idx.sum() == 0:
            acc = np.nan
        else:
            acc = float((y_pred[idx] == y_true[idx]).mean())
        out.append({"bin": name, "bin_acc": acc, "count": int(idx.sum())})
    return pd.DataFrame(out)


def _calibrate_tau(val_df: pd.DataFrame, probs: np.ndarray, wrong_direction_target: float, tau_init: float) -> dict:
    y_true = (val_df["y_sign"].to_numpy() > 0).astype(int)
    pred = (probs >= 0.5).astype(int)
    conf = np.maximum(probs, 1.0 - probs)

    taus = np.linspace(0.5, 0.99, 50)
    best = None
    for tau in taus:
        keep = conf >= tau
        if keep.sum() == 0:
            continue
        wrong = float((pred[keep] != y_true[keep]).mean())
        coverage = float(keep.mean())
        score = (wrong, -coverage)
        record = {"tau": float(tau), "wrong_direction": wrong, "coverage": coverage, "score": score}
        if wrong <= wrong_direction_target:
            if best is None or coverage > best["coverage"]:
                best = record
        elif best is None:
            best = record

    if best is None:
        best = {"tau": float(tau_init), "wrong_direction": 1.0, "coverage": 0.0, "score": (1.0, 0.0)}

    return {"tau": float(best["tau"]), "wrong_direction": float(best["wrong_direction"]), "coverage": float(best["coverage"])}


def train_sign_model(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str | Path, config) -> tuple[str, str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sign = _prepare_sign_df(train_df)
    val_sign = _prepare_sign_df(val_df)

    train_ds = _to_dataset(train_sign, config, training=True)
    val_ds = _to_dataset(val_sign, config, training=False)

    model = _build_sign_model(config.train.image_size, config.train.sign_lr)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auroc", patience=2, mode="max", restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=config.train.sign_epochs, callbacks=callbacks, verbose=2)

    model_path = out_dir / "sign_model.keras"
    model.save(model_path)

    probs = model.predict(val_ds, verbose=0).reshape(-1)
    y_true = (val_sign["y_sign"].to_numpy() > 0).astype(int)
    y_hat = (probs >= 0.5).astype(int)

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y_true, probs)
    auroc = float(auc_metric.result().numpy())
    bal_acc = _balanced_accuracy(y_true, y_hat)

    tau_info = _calibrate_tau(
        val_df=val_sign,
        probs=probs,
        wrong_direction_target=config.train.wrong_direction_target,
        tau_init=config.tau_init,
    )

    tau_path = out_dir / "tau.json"
    with tau_path.open("w", encoding="utf-8") as f:
        json.dump(tau_info, f, indent=2)

    bin_df = _per_bin_sign_accuracy(val_sign, y_hat)
    top_df = pd.DataFrame(
        [
            {
                "metric": "auroc",
                "value": auroc,
            },
            {
                "metric": "balanced_acc",
                "value": bal_acc,
            },
            {
                "metric": "tau",
                "value": tau_info["tau"],
            },
            {
                "metric": "tau_wrong_direction",
                "value": tau_info["wrong_direction"],
            },
            {
                "metric": "tau_coverage",
                "value": tau_info["coverage"],
            },
        ]
    )
    bin_df = bin_df.rename(columns={"bin_acc": "value"})
    bin_df["metric"] = "bin_acc_" + bin_df["bin"].astype(str)
    bin_df = bin_df[["metric", "value", "count"]]
    top_df["count"] = len(val_sign)

    metrics_df = pd.concat([top_df, bin_df], ignore_index=True)
    metrics_path = out_dir / "sign_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    pred_df = val_sign[["image_path", "y_sign", "y_mag"]].copy()
    pred_df["sign_prob"] = probs
    pred_df["confidence"] = np.maximum(probs, 1.0 - probs)
    pred_df["pred_sign"] = np.where(pred_df["sign_prob"] >= 0.5, 1, -1)
    pred_path = out_dir / "val_sign_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print(f"[DONE] Sign model: {model_path}")
    print(f"[DONE] Tau config: {tau_path}")
    print(f"[DONE] Sign metrics: {metrics_path}")
    print(f"[DONE] Sign val predictions: {pred_path}")

    return str(model_path), str(tau_path), str(metrics_path)
