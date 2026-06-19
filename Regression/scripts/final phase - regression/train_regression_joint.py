#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("TensorFlow is required for train_regression_joint.py") from exc


def _gaussian_kernel(size: int, sigma: float) -> tf.Tensor:
    x = tf.range(size, dtype=tf.float32) - (size - 1) / 2.0
    g = tf.exp(-(x ** 2) / (2.0 * sigma ** 2))
    g = g / tf.reduce_sum(g)
    kernel_2d = tf.tensordot(g, g, axes=0)
    return kernel_2d[:, :, tf.newaxis, tf.newaxis]


def _gaussian_blur(gray: tf.Tensor, sigma: float) -> tf.Tensor:
    ksize = int(max(3, 2 * int(3 * sigma) + 1))
    kernel = _gaussian_kernel(ksize, sigma)
    return tf.nn.depthwise_conv2d(gray, tf.tile(kernel, [1, 1, 1, 1]), strides=[1, 1, 1, 1], padding="SAME")


def _hf_energy(gray: tf.Tensor) -> tf.Tensor:
    # tf.image.sobel_edges expects rank-4 input [B,H,W,C].
    if gray.shape.rank == 3:
        gray4 = gray[tf.newaxis, ...]
    elif gray.shape.rank == 4:
        gray4 = gray
    else:
        raise ValueError(f"_hf_energy expects rank-3/4 tensor, got rank={gray.shape.rank}")

    sob = tf.image.sobel_edges(gray4)
    mag = tf.sqrt(tf.reduce_sum(tf.square(sob), axis=-1))
    return tf.reduce_mean(mag)


def _build_stage_b_inputs(img: tf.Tensor, image_size: int, sigma1: float, sigma2: float, sigma3: float):
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.image.convert_image_dtype(img, tf.float32)
    gray = tf.image.rgb_to_grayscale(img)

    blur1 = _gaussian_blur(gray[tf.newaxis, ...], sigma1)[0]
    blur2 = _gaussian_blur(gray[tf.newaxis, ...], sigma2)[0]
    blur3 = _gaussian_blur(gray[tf.newaxis, ...], sigma3)[0]

    d1 = gray - blur1
    d2 = blur1 - blur2
    d3 = blur2 - blur3

    i_ch = tf.clip_by_value(gray, 0.0, 1.0)
    d1_ch = tf.clip_by_value(d1 + 0.5, 0.0, 1.0)
    d2_ch = tf.clip_by_value(d2 + 0.5, 0.0, 1.0)
    img3 = tf.concat([i_ch, d1_ch, d2_ch], axis=-1)

    ehf = _hf_energy(tf.clip_by_value(d3 + 0.5, 0.0, 1.0))
    return img3, ehf


def _triplet_loss(embeddings: tf.Tensor, labels: tf.Tensor, margin: float) -> tf.Tensor:
    labels = tf.cast(labels, tf.int32)
    pdist = tf.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)
    same = tf.equal(labels[:, None], labels[None, :])
    diff = tf.logical_not(same)

    anchor_positive = tf.cast(same, tf.float32) - tf.eye(tf.shape(labels)[0], dtype=tf.float32)
    anchor_negative = tf.cast(diff, tf.float32)

    ap_dist = pdist[:, :, None]
    an_dist = pdist[:, None, :]
    losses = ap_dist - an_dist + margin

    mask = anchor_positive[:, :, None] * anchor_negative[:, None, :]
    losses = tf.maximum(losses * mask, 0.0)

    valid = tf.cast(losses > 1e-12, tf.float32)
    denom = tf.reduce_sum(valid)
    return tf.reduce_sum(losses) / tf.maximum(denom, 1.0)


class JointRegressor(tf.keras.Model):
    def __init__(self, image_size: int, lr: float, huber_delta: float, triplet_margin: float, triplet_weight: float):
        super().__init__()
        self.backbone = tf.keras.applications.MobileNetV3Small(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights=None,
            pooling="avg",
        )
        self.fusion = tf.keras.layers.Dense(128, activation="relu")
        self.embed_head = tf.keras.layers.Dense(32, name="embedding")
        self.reg_head = tf.keras.layers.Dense(1, activation="softplus", name="regression")

        self.huber = tf.keras.losses.Huber(delta=huber_delta)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.triplet_margin = triplet_margin
        self.triplet_weight = triplet_weight

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.huber_tracker = tf.keras.metrics.Mean(name="huber_loss")
        self.triplet_tracker = tf.keras.metrics.Mean(name="triplet_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.huber_tracker, self.triplet_tracker]

    def call(self, inputs, training=False):
        x = self.backbone(inputs["img"], training=training)
        ehf = tf.expand_dims(inputs["ehf"], axis=-1)
        x = tf.concat([x, ehf], axis=-1)
        x = self.fusion(x)
        emb = tf.math.l2_normalize(self.embed_head(x), axis=-1)
        reg = self.reg_head(x)
        return reg, emb

    def train_step(self, data):
        x, y = data
        y = tf.cast(y, tf.float32)
        with tf.GradientTape() as tape:
            reg, emb = self(x, training=True)
            reg = tf.squeeze(reg, axis=-1)
            huber = self.huber(y, reg)
            bins = tf.cast(tf.floor(y / 0.5), tf.int32)
            tri = _triplet_loss(emb, bins, self.triplet_margin)
            loss = huber + self.triplet_weight * tri

        grads = tape.gradient(loss, self.trainable_variables)
        grads = [
            tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else None
            for g in grads
        ]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.huber_tracker.update_state(huber)
        self.triplet_tracker.update_state(tri)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y = tf.cast(y, tf.float32)
        reg, emb = self(x, training=False)
        reg = tf.squeeze(reg, axis=-1)
        huber = self.huber(y, reg)
        bins = tf.cast(tf.floor(y / 0.5), tf.int32)
        tri = _triplet_loss(emb, bins, self.triplet_margin)
        loss = huber + self.triplet_weight * tri

        self.loss_tracker.update_state(loss)
        self.huber_tracker.update_state(huber)
        self.triplet_tracker.update_state(tri)
        return {m.name: m.result() for m in self.metrics}


def _branch_df(df: pd.DataFrame, sign: int) -> pd.DataFrame:
    branch = df[df["y_sign"] == sign].copy()
    if branch.empty:
        return branch
    branch["y_mag"] = pd.to_numeric(branch["y_mag"], errors="coerce")
    branch["y_mag"] = branch["y_mag"].astype(float)
    branch = branch.dropna(subset=["y_mag", "image_path"]).reset_index(drop=True)
    branch = branch[np.isfinite(branch["y_mag"])].copy().reset_index(drop=True)
    return branch


def _to_dataset(df: pd.DataFrame, config, training: bool) -> tf.data.Dataset:
    paths = df["image_path"].astype(str).values
    y = df["y_mag"].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    def _map(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img3, ehf = _build_stage_b_inputs(
            img,
            image_size=config.train.image_size,
            sigma1=config.sigma1,
            sigma2=config.sigma2,
            sigma3=config.sigma3,
        )
        features = {"img": img3, "ehf": ehf}
        return features, label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(4096, len(df)), seed=config.split_seed, reshuffle_each_iteration=True)
    ds = ds.batch(config.train.reg_batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _train_one_branch(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_path: Path,
    config,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    if len(train_df) < config.train.min_rows_per_branch:
        raise ValueError(
            f"Insufficient training rows for branch model: {len(train_df)} < {config.train.min_rows_per_branch}"
        )

    train_ds = _to_dataset(train_df, config, training=True)
    val_ds = _to_dataset(val_df, config, training=False)

    model = JointRegressor(
        image_size=config.train.image_size,
        lr=config.train.reg_lr,
        huber_delta=config.train.huber_delta,
        triplet_margin=config.train.triplet_margin,
        triplet_weight=config.train.triplet_weight,
    )
    model.compile(run_eagerly=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=config.train.reg_epochs, callbacks=callbacks, verbose=2)

    reg_pred, _ = model.predict(val_ds, verbose=0)
    reg_pred = reg_pred.reshape(-1)
    y_true = val_df["y_mag"].to_numpy()

    mae = float(np.mean(np.abs(reg_pred - y_true)))
    rmse = float(np.sqrt(np.mean((reg_pred - y_true) ** 2)))

    bins = [0.0, 0.5, 1.0, 2.0, np.inf]
    labels = ["(0,0.5]", "(0.5,1.0]", "(1.0,2.0]", "(2.0,inf]"]
    rows = [
        {"metric": "mae", "value": mae, "count": len(val_df)},
        {"metric": "rmse", "value": rmse, "count": len(val_df)},
    ]

    for lo, hi, name in zip(bins[:-1], bins[1:], labels):
        idx = (y_true > lo) & (y_true <= hi)
        if idx.sum() == 0:
            b_mae, b_rmse = np.nan, np.nan
        else:
            b_mae = float(np.mean(np.abs(reg_pred[idx] - y_true[idx])))
            b_rmse = float(np.sqrt(np.mean((reg_pred[idx] - y_true[idx]) ** 2)))
        rows.append({"metric": f"bin_mae_{name}", "value": b_mae, "count": int(idx.sum())})
        rows.append({"metric": f"bin_rmse_{name}", "value": b_rmse, "count": int(idx.sum())})

    pred_df = val_df[["image_path"]].copy()
    pred_df["y_mag_true"] = y_true
    pred_df["y_mag_pred"] = reg_pred

    model.save(out_path)
    return out_path, pd.DataFrame(rows), pred_df


def train_regressors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: str | Path,
    config,
) -> tuple[str, str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plus_train = _branch_df(train_df, sign=1)
    plus_val = _branch_df(val_df, sign=1)
    minus_train = _branch_df(train_df, sign=-1)
    minus_val = _branch_df(val_df, sign=-1)

    r_plus_path = out_dir / "R_plus.keras"
    r_minus_path = out_dir / "R_minus.keras"

    trained_plus = False
    trained_minus = False
    plus_metrics = pd.DataFrame()
    minus_metrics = pd.DataFrame()
    plus_pred = pd.DataFrame(columns=["image_path", "y_mag_true", "y_mag_pred"])
    minus_pred = pd.DataFrame(columns=["image_path", "y_mag_true", "y_mag_pred"])

    if not plus_train.empty and not plus_val.empty:
        r_plus_path, plus_metrics, plus_pred = _train_one_branch(plus_train, plus_val, r_plus_path, config)
        trained_plus = True
    else:
        print("[WARN] No positive-sign samples for R_plus. Will reuse R_minus model if available.")

    if not minus_train.empty and not minus_val.empty:
        r_minus_path, minus_metrics, minus_pred = _train_one_branch(minus_train, minus_val, r_minus_path, config)
        trained_minus = True
    else:
        print("[WARN] No negative-sign samples for R_minus. Will reuse R_plus model if available.")

    if not trained_plus and not trained_minus:
        raise ValueError("No valid data available to train either regression branch.")

    # If one branch is missing, copy the trained branch model as fallback to keep pipeline runnable.
    if trained_plus and not trained_minus:
        import shutil

        shutil.copy2(r_plus_path, r_minus_path)
        minus_metrics = pd.DataFrame([{"metric": "fallback_from_plus", "value": 1.0, "count": 0}])
        minus_pred = plus_pred.rename(columns={"y_mag_pred": "y_mag_pred"}).copy()
        trained_minus = True
    elif trained_minus and not trained_plus:
        import shutil

        shutil.copy2(r_minus_path, r_plus_path)
        plus_metrics = pd.DataFrame([{"metric": "fallback_from_minus", "value": 1.0, "count": 0}])
        plus_pred = minus_pred.rename(columns={"y_mag_pred": "y_mag_pred"}).copy()
        trained_plus = True

    if not plus_metrics.empty:
        plus_metrics["branch"] = "plus"
    if not minus_metrics.empty:
        minus_metrics["branch"] = "minus"
    metrics_df = pd.concat([plus_metrics, minus_metrics], ignore_index=True)

    metrics_path = out_dir / "reg_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    plus_pred = plus_pred.rename(columns={"y_mag_pred": "mag_pred_plus"})
    minus_pred = minus_pred.rename(columns={"y_mag_pred": "mag_pred_minus"})
    pred_df = plus_pred[["image_path", "mag_pred_plus"]].merge(
        minus_pred[["image_path", "mag_pred_minus"]],
        on="image_path",
        how="outer",
    )
    pred_path = out_dir / "val_reg_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print(f"[DONE] Regression model (+): {r_plus_path}")
    print(f"[DONE] Regression model (-): {r_minus_path}")
    print(f"[DONE] Regression metrics: {metrics_path}")
    print(f"[DONE] Regression val predictions: {pred_path}")

    return str(r_plus_path), str(r_minus_path), str(metrics_path)
