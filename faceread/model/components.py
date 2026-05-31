# -*- coding: utf-8 -*-
"""
model/components.py
Main Quest ✅:
  - Custom Layer    : ChannelAttention (Squeeze-and-Excitation)
  - Custom Loss     : FocalLoss
  - Custom Callback : OverfittingMonitor
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras as keras_core


# ── Custom Layer ──────────────────────────────────────────────
@keras_core.saving.register_keras_serializable(package="FER")
class ChannelAttention(layers.Layer):
    """
    Squeeze-and-Excitation Channel Attention.
    Main Quest: Custom Layer ✅
    """

    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap     = layers.GlobalAveragePooling2D()
        self.dense1  = layers.Dense(
            max(1, num_channels // self.reduction_ratio),
            activation='relu', use_bias=False
        )
        self.dense2  = layers.Dense(num_channels, activation='sigmoid', use_bias=False)
        self.reshape = layers.Reshape((1, 1, num_channels))
        super().build(input_shape)

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return inputs * x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'reduction_ratio': self.reduction_ratio})
        return cfg


# ── Custom Loss ───────────────────────────────────────────────
@keras_core.saving.register_keras_serializable(package="FER")
class FocalLoss(keras.losses.Loss):
    """
    Focal Loss — bantu kelas minority.
    Main Quest: Custom Loss Function ✅
    """

    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred       = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce           = -y_true * tf.math.log(y_pred)
        p_t          = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = self.alpha * tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * ce, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'alpha': self.alpha, 'gamma': self.gamma})
        return cfg


# ── Custom Callback ───────────────────────────────────────────
class OverfittingMonitor(keras.callbacks.Callback):
    """
    Monitor gap train vs val accuracy/loss setiap epoch.
    Main Quest: Custom Callback ✅
    """

    def __init__(self, acc_gap_threshold=0.10, loss_gap_threshold=0.30):
        super().__init__()
        self.acc_gap_threshold  = acc_gap_threshold
        self.loss_gap_threshold = loss_gap_threshold
        self.overfit_epochs     = []

    def on_epoch_end(self, epoch, logs=None):
        acc_gap  = logs.get("accuracy", 0) - logs.get("val_accuracy", 0)
        loss_gap = logs.get("val_loss",  0) - logs.get("loss", 0)
        is_overfit = (acc_gap > self.acc_gap_threshold
                      or loss_gap > self.loss_gap_threshold)
        status = "⚠️  OVERFITTING" if is_overfit else "✅ OK"
        if is_overfit:
            self.overfit_epochs.append(epoch + 1)
        print(f"  [OvfMonitor] Ep {epoch+1:02d} | "
              f"acc_gap={acc_gap:.4f} | loss_gap={loss_gap:.4f} | {status}")

    def on_train_end(self, logs=None):
        if self.overfit_epochs:
            print(f"\n[OvfMonitor] Overfitting di epoch: {self.overfit_epochs}")
        else:
            print("\n[OvfMonitor] Tidak ada indikasi overfitting signifikan.")
            