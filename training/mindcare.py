# -*- coding: utf-8 -*-
"""
facial_expression_resnet50_vscode.py
ResNet50 Fine-tuning — 4 kelas: angry, happy, sad, neutral
Target: 85%+ accuracy

Jalankan:
    python facial_expression_resnet50_vscode.py

Setelah selesai:
    Terminal 1: uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload
    Terminal 2: python inference/inference_camera.py
    TensorBoard: tensorboard --logdir logs/fit
"""

import os
import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras as keras_core
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

print("GPU tersedia   :", tf.config.list_physical_devices("GPU"))
print("CPU tersedia   :", tf.config.list_physical_devices("CPU"))
print("TensorFlow ver :", tf.__version__)


# ============================================================
# KONFIGURASI
# ============================================================

TRAIN_DIR   = "Dataset/train"
TEST_DIR    = "Dataset/test"
IMG_SIZE    = 48
TARGET_SIZE = 224
BATCH_SIZE  = 16
EPOCHS_FT   = 30
PATIENCE    = 12

CLASS_NAMES   = ["angry", "happy", "sad", "neutral"]
NUM_CLASSES   = len(CLASS_NAMES)
TEXT_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}

os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs/fit",     exist_ok=True)


# ============================================================
# CUSTOM COMPONENTS
# ============================================================

@keras_core.saving.register_keras_serializable(package="FER")
class CustomCrossEntropy(keras.losses.Loss):
    """Label Smoothing Cross-Entropy loss."""

    def __init__(self, smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, label_smoothing=self.smoothing
        )

    def get_config(self):
        return {**super().get_config(), "smoothing": self.smoothing}


class OverfittingMonitor(Callback):
    """Monitor gap accuracy train vs val setiap epoch."""

    def __init__(self, acc_gap_threshold=0.10):
        super().__init__()
        self.acc_gap_threshold = acc_gap_threshold
        self.overfit_epochs    = []

    def on_epoch_end(self, epoch, logs=None):
        acc_gap  = logs.get("accuracy", 0) - logs.get("val_accuracy", 0)
        loss_gap = logs.get("val_loss",  0) - logs.get("loss", 0)
        status   = "OVERFITTING" if acc_gap > self.acc_gap_threshold else "OK"
        if acc_gap > self.acc_gap_threshold:
            self.overfit_epochs.append(epoch + 1)
        print(
            f"  [Monitor] Ep {epoch+1:02d} | "
            f"acc_gap={acc_gap:.4f} | loss_gap={loss_gap:.4f} | {status}"
        )

    def on_train_end(self, logs=None):
        if self.overfit_epochs:
            print(f"\n[Monitor] Overfit epochs: {self.overfit_epochs}")
        else:
            print("\n[Monitor] Tidak ada overfitting signifikan.")


# ============================================================
# FUNGSI DATASET
# ============================================================

def load_images_from_folder(folder_path, class_names, img_size=48):
    """Muat gambar grayscale dari folder per kelas."""
    X, y = [], []
    for cls in class_names:
        cls_path = os.path.join(folder_path, cls)
        if not os.path.exists(cls_path):
            print(f"[WARNING] Folder tidak ditemukan: {cls_path}")
            continue
        label, loaded = TEXT_TO_LABEL[cls], 0
        for fname in os.listdir(cls_path):
            img = cv2.imread(os.path.join(cls_path, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(cv2.resize(img, (img_size, img_size)))
            y.append(label)
            loaded += 1
        print(f"  [{cls}] {loaded} gambar dimuat")
    return np.array(X, dtype="float32"), np.array(y, dtype="int32")


def make_dataset(X_flat, y_labels, img_size=48, target_size=224,
                 batch_size=32, shuffle=False, augment=False):
    """
    Buat tf.data.Dataset dari array flat.
    augment=True hanya untuk split train; val & test tidak di-augmentasi.
    """
    def parse_fn(x_flat, label):
        img = tf.reshape(x_flat, (img_size, img_size, 1))
        img = tf.image.grayscale_to_rgb(img)
        img = tf.image.resize(img, (target_size, target_size))
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, tf.one_hot(label, NUM_CLASSES)

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices(
        (X_flat.astype("float32"), y_labels.astype("int32"))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=4096, seed=42)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================
# FUNGSI MODEL
# ============================================================

def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Bangun model dengan Functional API.
    Arsitektur:
        Input → Augmentation → ResNet50 Backbone
        → CNN Block 1 (256) → CNN Block 2 (128)
        → GlobalAvgPool → Dense(512) → Dense(256) → Dense(4, softmax)
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Augmentation (aktif otomatis hanya saat training=True)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(factor=0.05, fill_mode="reflect")(x)
    x = layers.RandomZoom(height_factor=(-0.08, 0.08), fill_mode="reflect")(x)
    x = layers.RandomTranslation(height_factor=0.07, width_factor=0.07, fill_mode="reflect")(x)

    # ResNet50 backbone — frozen pada Fase 1
    backbone = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    backbone.trainable = False
    x = backbone(x, training=False)  # BN backbone selalu inference mode

    # CNN Block 1
    x = layers.Conv2D(256, (3, 3), padding="same",
                      kernel_regularizer=keras.regularizers.l2(2e-4),
                      name="cnn_block1_conv")(x)
    x = layers.BatchNormalization(name="cnn_block1_bn")(x)
    x = layers.Activation("relu",   name="cnn_block1_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_block1_maxpool")(x)

    # CNN Block 2
    x = layers.Conv2D(128, (3, 3), padding="same",
                      kernel_regularizer=keras.regularizers.l2(2e-4),
                      name="cnn_block2_conv")(x)
    x = layers.BatchNormalization(name="cnn_block2_bn")(x)
    x = layers.Activation("relu",   name="cnn_block2_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_block2_maxpool")(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(2e-4), name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_512")(x)
    x = layers.Dropout(0.65, name="dropout_512")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(2e-4), name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Dropout(0.55, name="dropout_256")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs=inputs, outputs=outputs, name="FER_ResNet50_FunctionalAPI"), backbone


# ============================================================
# FUNGSI VISUALISASI
# ============================================================

def plot_confusion_matrix(y_true, y_pred, class_names, test_acc):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix | Accuracy: {test_acc*100:.2f}%")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
    print("Confusion matrix disimpan ke confusion_matrix.png")


def plot_overfitting_analysis(history_phase1, history_ft):
    acc      = history_phase1.history["accuracy"]     + history_ft["accuracy"]
    val_acc  = history_phase1.history["val_accuracy"] + history_ft["val_accuracy"]
    loss     = history_phase1.history["loss"]         + history_ft["loss"]
    val_loss = history_phase1.history["val_loss"]     + history_ft["val_loss"]
    epochs   = range(1, len(acc) + 1)
    split    = len(history_phase1.history["accuracy"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Overfitting Analysis — ResNet50 Functional API",
                 fontsize=16, fontweight="bold")

    # Accuracy
    axes[0, 0].plot(epochs, acc,     "b-o", markersize=3, label="Train")
    axes[0, 0].plot(epochs, val_acc, "r-o", markersize=3, label="Val")
    axes[0, 0].fill_between(epochs, acc, val_acc, alpha=0.15, color="orange")
    axes[0, 0].axvline(x=split, color="purple", linestyle="--", label="Fine-tune start")
    axes[0, 0].set_title("Accuracy"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.4)

    # Loss
    axes[0, 1].plot(epochs, loss,     "b-o", markersize=3, label="Train")
    axes[0, 1].plot(epochs, val_loss, "r-o", markersize=3, label="Val")
    axes[0, 1].fill_between(epochs, loss, val_loss, alpha=0.15, color="orange")
    axes[0, 1].axvline(x=split, color="purple", linestyle="--", label="Fine-tune start")
    axes[0, 1].set_title("Loss"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.4)

    # Accuracy gap
    acc_gap = [a - v for a, v in zip(acc, val_acc)]
    colors  = ["red" if g > 0.10 else "green" for g in acc_gap]
    axes[1, 0].bar(epochs, acc_gap, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=0.10, color="red", linestyle="--", label="Threshold")
    axes[1, 0].set_title("Accuracy Gap"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.4, axis="y")

    # Loss ratio
    ratio  = [vl / tl if tl > 0 else 1 for tl, vl in zip(loss, val_loss)]
    rcolor = ["red" if r > 1.3 else "steelblue" for r in ratio]
    axes[1, 1].plot(epochs, ratio, "k-", linewidth=1.5, alpha=0.5)
    axes[1, 1].scatter(epochs, ratio, c=rcolor, s=30, zorder=5)
    axes[1, 1].axhline(y=1.30, color="red",   linestyle="--", label="Threshold")
    axes[1, 1].axhline(y=1.0,  color="green", linestyle="--", label="Ideal")
    axes[1, 1].set_title("Loss Ratio"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig("overfitting_analysis.png", dpi=150, bbox_inches="tight")
    print("Overfitting analysis disimpan ke overfitting_analysis.png")


# ============================================================
# MAIN
# ============================================================

def main():
    # ----------------------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------------------
    print("Loading training data...")
    X_train_raw, y_train_raw = load_images_from_folder(TRAIN_DIR, CLASS_NAMES, IMG_SIZE)
    print(f"Train raw: {X_train_raw.shape}")

    print("\nLoading test data...")
    X_test_raw, y_test_raw = load_images_from_folder(TEST_DIR, CLASS_NAMES, IMG_SIZE)
    print(f"Test raw: {X_test_raw.shape}")

    # ----------------------------------------------------------
    # 2. Oversampling (hanya train)
    # ----------------------------------------------------------
    X_flat = X_train_raw.reshape(-1, IMG_SIZE * IMG_SIZE)
    X_resampled, y_resampled = RandomOverSampler(
        sampling_strategy="auto", random_state=42
    ).fit_resample(X_flat, y_train_raw)
    print(f"\nAfter oversampling: {X_resampled.shape}")

    # ----------------------------------------------------------
    # 3. Split & buat tf.data pipeline
    # ----------------------------------------------------------
    X_train_flat, X_val_flat, y_train_lbl, y_val_lbl = train_test_split(
        X_resampled, y_resampled, test_size=0.1, random_state=45, stratify=y_resampled
    )

    print("\nMembuat tf.data pipeline...")
    train_ds = make_dataset(X_train_flat, y_train_lbl, shuffle=True,  augment=True,  batch_size=BATCH_SIZE)
    val_ds   = make_dataset(X_val_flat,   y_val_lbl,   shuffle=False, augment=False, batch_size=BATCH_SIZE)
    test_ds  = make_dataset(
        X_test_raw.reshape(-1, IMG_SIZE * IMG_SIZE), y_test_raw,
        shuffle=False, augment=False, batch_size=BATCH_SIZE
    )
    print(f"Train batches : {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ----------------------------------------------------------
    # 4. Build model
    # ----------------------------------------------------------
    model, backbone = build_model()

    with open("model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    model.summary()
    print(f"Total params: {model.count_params():,}")
    print("Model summary disimpan ke model_summary.txt")

    # ----------------------------------------------------------
    # 5. Fase 1 — Train classifier head (backbone frozen)
    # ----------------------------------------------------------
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"TensorBoard log dir: {log_dir}")

    callbacks_phase1 = [
        OverfittingMonitor(acc_gap_threshold=0.10),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="saved_models/best_model.keras",
            monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch"
        ),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=CustomCrossEntropy(smoothing=0.1),
        metrics=["accuracy"],
    )

    print("\n" + "=" * 60)
    print("FASE 1: Training Classifier Head (20 epoch, ResNet frozen)")
    print("=" * 60)
    history_phase1 = model.fit(
        train_ds, epochs=20, validation_data=val_ds,
        callbacks=callbacks_phase1, verbose=1,
    )
    print(f"Fase 1 selesai! Best val_acc: {max(history_phase1.history['val_accuracy']):.4f}")

    # ----------------------------------------------------------
    # 6. Fase 2 — Fine-tuning dengan tf.GradientTape
    # ----------------------------------------------------------
    # Unfreeze 15 layer terakhir backbone (BN tetap frozen)
    backbone.trainable = True
    for layer in backbone.layers[:-15]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    print(f"Trainable params setelah unfreeze: "
          f"{sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

    steps_per_epoch = len(train_ds)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-6,
        decay_steps=steps_per_epoch * EPOCHS_FT,
        alpha=1e-7,
    )
    optimizer        = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn          = CustomCrossEntropy(smoothing=0.1)
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    val_acc_metric   = keras.metrics.CategoricalAccuracy()
    val_loss_metric  = keras.metrics.Mean()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss   = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, logits)
        return loss

    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        val_acc_metric.update_state(y, logits)
        val_loss_metric.update_state(loss_fn(y, logits))

    writer = tf.summary.create_file_writer(
        "logs/fit/ft_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    best_val_acc = 0.0
    best_weights = None
    patience_cnt = 0
    history_ft   = {"loss": [], "accuracy": [], "val_accuracy": [], "val_loss": []}

    print("\n" + "=" * 60)
    print("FASE 2: Fine-tuning dengan tf.GradientTape")
    print("=" * 60)

    for epoch in range(EPOCHS_FT):
        losses    = [float(train_step(x, y)) for x, y in train_ds]
        for x, y in val_ds:
            val_step(x, y)

        avg_loss  = np.mean(losses)
        train_acc = float(train_acc_metric.result())
        val_acc   = float(val_acc_metric.result())
        val_loss  = float(val_loss_metric.result())

        with writer.as_default():
            tf.summary.scalar("ft/train_acc", train_acc, step=epoch)
            tf.summary.scalar("ft/val_acc",   val_acc,   step=epoch)
            tf.summary.scalar("ft/val_loss",  val_loss,  step=epoch)

        history_ft["loss"].append(avg_loss)
        history_ft["accuracy"].append(train_acc)
        history_ft["val_accuracy"].append(val_acc)
        history_ft["val_loss"].append(val_loss)

        status = "OVERFIT" if train_acc - val_acc > 0.10 else "OK"
        print(
            f"Ep {epoch+1:02d}/{EPOCHS_FT} | loss={avg_loss:.4f} | "
            f"acc={train_acc:.4f} | val_acc={val_acc:.4f} | {status}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()
            model.save("facial_expression_model.keras")
            model.save("saved_models/best_model.keras")
            print(f"  Best saved! val_acc={val_acc*100:.2f}%")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping epoch {epoch+1}")
                model.set_weights(best_weights)
                print(f"  Bobot dikembalikan ke best val_acc={best_val_acc*100:.2f}%")
                break

        train_acc_metric.reset_state()
        val_acc_metric.reset_state()
        val_loss_metric.reset_state()

    print(f"\nBest val_accuracy: {best_val_acc*100:.2f}%")

    # ----------------------------------------------------------
    # 7. Evaluasi
    # ----------------------------------------------------------
    print("\n=== EVALUASI PADA TEST SET ===")
    model = keras.models.load_model(
        "facial_expression_model.keras",
        custom_objects={"CustomCrossEntropy": CustomCrossEntropy},
    )
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Target 85%    : {'TERCAPAI!' if test_acc >= 0.85 else 'Belum tercapai'}")

    y_pred = np.argmax(model.predict(test_ds), axis=1)
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test_raw, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(y_test_raw, y_pred, CLASS_NAMES, test_acc)
    plot_overfitting_analysis(history_phase1, history_ft)

    # ----------------------------------------------------------
    # 8. Simpan model final
    # ----------------------------------------------------------
    model.save("facial_expression_model.keras")
    model.export("facial_expression_savedmodel")

    print("\n" + "=" * 60)
    print(f"Training selesai! Test Accuracy: {test_acc*100:.2f}%")
    print("=" * 60)
    print("\nFile yang dihasilkan:")
    print("  facial_expression_model.keras   ← Model final")
    print("  facial_expression_savedmodel/   ← TF SavedModel")
    print("  saved_models/best_model.keras   ← Best checkpoint")
    print("  confusion_matrix.png")
    print("  overfitting_analysis.png")


if __name__ == "__main__":
    main()