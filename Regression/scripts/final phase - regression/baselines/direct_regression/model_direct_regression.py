#!/usr/bin/env python3
from __future__ import annotations

import tensorflow as tf


def _inv_res_block(x, out_ch: int, stride: int, expand: int):
    in_ch = int(x.shape[-1])
    hidden = max(in_ch * int(expand), 8)

    y = x
    if expand != 1:
        y = tf.keras.layers.Conv2D(hidden, 1, padding='same', use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.ReLU(max_value=6.0)(y)

    y = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU(max_value=6.0)(y)

    y = tf.keras.layers.Conv2D(out_ch, 1, padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)

    if stride == 1 and in_ch == out_ch:
        y = tf.keras.layers.Add()([x, y])
    return y


def build_direct_regression_model(input_shape: tuple[int, int, int] = (200, 200, 4)) -> tf.keras.Model:
    """Lightweight signed-distance regressor using the same 4-channel ROI input scope as Phase 5."""
    inp = tf.keras.Input(shape=input_shape, name='XB')

    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    x = _inv_res_block(x, out_ch=16, stride=1, expand=1)
    x = _inv_res_block(x, out_ch=24, stride=2, expand=4)
    x = _inv_res_block(x, out_ch=24, stride=1, expand=3)
    x = _inv_res_block(x, out_ch=40, stride=2, expand=3)
    x = _inv_res_block(x, out_ch=40, stride=1, expand=3)
    x = _inv_res_block(x, out_ch=64, stride=2, expand=3)
    x = _inv_res_block(x, out_ch=64, stride=1, expand=3)

    x = tf.keras.layers.Conv2D(96, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(gap)
    x = tf.keras.layers.Dropout(0.15)(x)
    y_hat = tf.keras.layers.Dense(1, activation='linear', dtype='float32', name='signed_dz_um')(x)
    return tf.keras.Model(inputs=inp, outputs=y_hat, name='direct_signed_regressor')
