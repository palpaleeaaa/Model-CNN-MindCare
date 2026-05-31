# -*- coding: utf-8 -*-
"""
model/builder.py
Membangun arsitektur model menggunakan TF Functional API.
Main Quest: TensorFlow Functional API ✅
"""

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

from config import IMG_SIZE, NUM_CLASSES


def build_model(freeze_base: bool = True) -> tuple[Model, Model]:
    """
    Bangun model FER dengan MobileNetV2 sebagai backbone.
    Main Quest: TensorFlow Functional API ✅

    Returns:
        model      : full model siap training
        base_model : referensi ke backbone (untuk unfreeze di phase 2)
    """
    # Augmentasi — hanya aktif saat training=True
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.10),
        layers.RandomBrightness((-0.15, 0.15)),
    ], name="data_augmentation")

    # Backbone pretrained
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = not freeze_base

    # Functional API ─────────────────────────────────────────
    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x       = data_augmentation(inputs)
    x       = base_model(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.30)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.20)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="FER_MobileNetV2")
    return model, base_model