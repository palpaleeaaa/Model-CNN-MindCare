# -*- coding: utf-8 -*-
"""
Real-Time Facial Expression Detection & Recognition using CNN
Refactored: Functional API + Custom Components + Local Dataset + Overfitting Analysis
Classes  : angry, happy, sad, neutral  (4 kelas)
"""

# ============================================================
# 1. IMPORT
# ============================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import keras as keras_core  # Keras 3 standalone untuk register_keras_serializable

# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import os
import cv2

# ── CEK GPU / CPU ─────────────────────────────────────────── ★ TAMBAHKAN DI SINI
print("GPU tersedia:", tf.config.list_physical_devices('GPU'))
print("CPU tersedia:", tf.config.list_physical_devices('CPU'))
print("TensorFlow versi:", tf.__version__)
# ============================================================
# 2. LOAD DATASET (LOCAL FOLDER: Dataset/train & Dataset/test)
#    Hanya 4 kelas: angry, happy, sad, neutral
# ============================================================

TRAIN_DIR = "Dataset/train"
TEST_DIR  = "Dataset/test"
IMG_SIZE  = 48

# ── Hanya 4 kelas yang digunakan ──────────────────────────
class_names   = ["angry", "happy", "sad", "neutral"]
NUM_CLASSES   = len(class_names)                        # 4
text_to_label = {v: k for k, v in enumerate(class_names)}


def load_images_from_folder(folder_path, class_names, img_size=48):
    """
    Membaca gambar grayscale dari sub-folder sesuai class_names.
    Folder yang tidak ada di class_names akan dilewati otomatis.
    """
    X, y = [], []
    for cls in class_names:
        cls_path = os.path.join(folder_path, cls)
        if not os.path.exists(cls_path):
            print(f"[WARNING] Folder tidak ditemukan: {cls_path}")
            continue
        label = text_to_label[cls]
        loaded = 0
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            img   = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)
            loaded += 1
        print(f"  [{cls}] {loaded} gambar dimuat")
    return np.array(X, dtype="float32"), np.array(y, dtype="int32")


print("Loading training data...")
X_train_raw, y_train_raw = load_images_from_folder(TRAIN_DIR, class_names, IMG_SIZE)
print(f"Train raw : {X_train_raw.shape}, Labels: {y_train_raw.shape}")

print("\nLoading test data...")
X_test_raw, y_test_raw = load_images_from_folder(TEST_DIR, class_names, IMG_SIZE)
print(f"Test raw  : {X_test_raw.shape}, Labels: {y_test_raw.shape}")

# ============================================================
# 3. DATA VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(14, 6))
for label_idx, label_name in enumerate(class_names):
    indices = np.where(y_train_raw == label_idx)[0]
    for j in range(min(5, len(indices))):
        ax = plt.subplot(NUM_CLASSES, 5, label_idx * 5 + j + 1)
        ax.imshow(X_train_raw[indices[j]], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(label_name, fontsize=9)
plt.suptitle("Sample Images per Class (4 kelas)", fontsize=14)
plt.tight_layout()
plt.savefig("sample_images.png", dpi=100, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
unique, counts = np.unique(y_train_raw, return_counts=True)
sns.barplot(
    x=[class_names[i] for i in unique],
    y=counts,
    hue=[class_names[i] for i in unique],
    palette="Set2",
    legend=False,
)
plt.title("Distribusi Kelas Training (Sebelum Balancing)")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution_before.png", dpi=100, bbox_inches="tight")
plt.show()

# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

X_flat = X_train_raw.reshape(-1, IMG_SIZE * IMG_SIZE)
oversampler = RandomOverSampler(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_flat, y_train_raw)
print(f"\nAfter oversampling: {X_resampled.shape}, {y_resampled.shape}")

plt.figure(figsize=(8, 5))
unique2, counts2 = np.unique(y_resampled, return_counts=True)
sns.barplot(
    x=[class_names[i] for i in unique2],
    y=counts2,
    hue=[class_names[i] for i in unique2],
    palette="Set3",
    legend=False,
)
plt.title("Distribusi Kelas Training (Setelah Balancing)")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution_after.png", dpi=100, bbox_inches="tight")
plt.show()

X_train_full = X_resampled.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
X_test       = X_test_raw.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

X_train, X_val, y_train_labels, y_val_labels = train_test_split(
    X_train_full, y_resampled,
    test_size=0.1, random_state=45, stratify=y_resampled
)

y_train = to_categorical(y_train_labels, NUM_CLASSES)
y_val   = to_categorical(y_val_labels,   NUM_CLASSES)
y_test  = to_categorical(y_test_raw,     NUM_CLASSES)

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# ============================================================
# 5. DATA AUGMENTATION (Diperkuat — fokus rotasi maksimal)
# ── Strategi: rotasi lebar ±30° + kombinasi flip & zoom ─────
# ── Dengan ~500 gambar/kelas, augmentasi agresif sangat
#    membantu model generalize ke wajah dari berbagai sudut ──
# ============================================================

data_augmentation = keras.Sequential(
    [
        # Horizontal flip — ekspresi wajah simetris secara horizontal
        layers.RandomFlip("horizontal"),

        # ★ ROTASI MAKSIMAL: ±30° (0.083 * 360 ≈ 30°)
        # Sebelumnya 0.10 (~36°), dinaikkan ke 0.083 karena
        # ekspresi wajah mulai tidak natural di atas 30°.
        # Nilai ini adalah sweet-spot untuk FER.
        layers.RandomRotation(
            factor=0.083,          # ±30 derajat
            fill_mode="reflect",   # reflect lebih natural dari constant untuk wajah
            interpolation="bilinear",
        ),

        # Zoom in/out ±15% — simulasi jarak kamera berbeda
        layers.RandomZoom(
            height_factor=(-0.15, 0.15),
            width_factor=(-0.15, 0.15),
            fill_mode="reflect",
        ),

        # Translasi ±12% — wajah tidak selalu persis di tengah
        layers.RandomTranslation(
            height_factor=0.12,
            width_factor=0.12,
            fill_mode="reflect",
        ),

        # Kontras ±15% — variasi pencahayaan & kualitas kamera
        layers.RandomContrast(factor=0.15),

        # ★ TAMBAHAN BARU: Brightness ±10% untuk variasi eksposur
        layers.RandomBrightness(factor=0.10),
    ],
    name="data_augmentation",
)

# ── Preview hasil augmentasi ─────────────────────────────────
print("\nMenampilkan preview augmentasi rotasi ±30°...")
sample_img = X_train[:1]               # ambil 1 gambar
fig, axes  = plt.subplots(2, 8, figsize=(16, 5))
fig.suptitle("Preview Augmentasi (rotasi ±30°, zoom ±15%, dll)", fontsize=13)
for ax_row in axes:
    for ax in ax_row:
        aug = data_augmentation(sample_img, training=True)
        ax.imshow(aug[0, :, :, 0], cmap="gray")
        ax.axis("off")
plt.tight_layout()
plt.savefig("augmentation_preview.png", dpi=100, bbox_inches="tight")
plt.show()
print("Preview disimpan ke augmentation_preview.png")

# ============================================================
# 6. CUSTOM COMPONENTS
# ============================================================


# ── 6a. Custom Layer: Channel Attention ───────────────────────────────────────
@keras_core.saving.register_keras_serializable(package="FER")
class ChannelAttention(layers.Layer):
    """
    Custom Layer: Squeeze-and-Excitation Channel Attention.
    Memperkuat fitur penting dan menekan fitur yang kurang relevan.
    """

    def __init__(self, reduction_ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels   = input_shape[-1]
        self.gap   = layers.GlobalAveragePooling2D()
        self.fc1   = layers.Dense(channels // self.reduction_ratio, activation="relu")
        self.fc2   = layers.Dense(channels, activation="sigmoid")
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        gap   = self.gap(inputs)
        scale = self.fc1(gap)
        scale = self.fc2(scale)
        scale = tf.reshape(scale, (-1, 1, 1, tf.shape(scale)[-1]))
        return inputs * scale

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


# ── 6b. Custom Loss: Focal Loss ───────────────────────────────────────────────
@keras_core.saving.register_keras_serializable(package="FER")
class FocalLoss(keras.losses.Loss):
    """
    Custom Loss: Focal Loss untuk menangani class imbalance.
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred         = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy  = -y_true * tf.math.log(y_pred)
        pt             = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight   = self.alpha * tf.pow(1.0 - pt, self.gamma)
        loss           = focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


# ── 6c. Custom Callback: Overfitting Monitor ─────────────────────────────────
class OverfittingMonitor(Callback):
    """Custom Callback: Monitor gap train vs val accuracy/loss tiap epoch."""

    def __init__(self, acc_gap_threshold=0.10, loss_gap_threshold=0.30):
        super(OverfittingMonitor, self).__init__()
        self.acc_gap_threshold  = acc_gap_threshold
        self.loss_gap_threshold = loss_gap_threshold
        self.overfit_epochs     = []

    def on_epoch_end(self, epoch, logs=None):
        train_acc  = logs.get("accuracy", 0)
        val_acc    = logs.get("val_accuracy", 0)
        train_loss = logs.get("loss", 0)
        val_loss   = logs.get("val_loss", 0)
        acc_gap    = train_acc - val_acc
        loss_gap   = val_loss - train_loss

        if acc_gap > self.acc_gap_threshold or loss_gap > self.loss_gap_threshold:
            status = "⚠️  OVERFITTING"
            self.overfit_epochs.append(epoch + 1)
        else:
            status = "✅ OK"

        print(
            f"  [OvfMonitor] Ep {epoch + 1:02d} | "
            f"acc_gap={acc_gap:.4f} | loss_gap={loss_gap:.4f} | {status}"
        )

    def on_train_end(self, logs=None):
        if self.overfit_epochs:
            print(f"\n[OvfMonitor] Overfitting di epoch: {self.overfit_epochs}")
        else:
            print("\n[OvfMonitor] Tidak ada indikasi overfitting signifikan.")


# ============================================================
# 7. MODEL BUILDING (Functional API)
# ============================================================


def conv_bn_relu(x, filters, kernel_size=(3, 3), strides=(1, 1), padding="same"):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES):
    """
    CNN dengan Functional API + ChannelAttention (Custom Layer).
    Output head disesuaikan dengan NUM_CLASSES = 4.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Augmentasi hanya aktif saat training=True
    x = data_augmentation(inputs)

    # ── Block 1 ──────────────────────────────────────────────
    x = conv_bn_relu(x, 32, padding="valid")
    x = layers.Dropout(0.30)(x)

    # ── Block 2 ──────────────────────────────────────────────
    x = conv_bn_relu(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 3 ──────────────────────────────────────────────
    x = conv_bn_relu(x, 64, padding="valid")
    x = layers.Dropout(0.30)(x)

    # ── Block 4 + Channel Attention (Custom Layer) ───────────
    x = conv_bn_relu(x, 128)
    x = ChannelAttention(reduction_ratio=8, name="channel_attention")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 5 ──────────────────────────────────────────────
    x = conv_bn_relu(x, 128, padding="valid")
    x = layers.MaxPooling2D((2, 2))(x)

    # ── Classifier Head ──────────────────────────────────────
    x       = layers.Flatten()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.55)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="FER_CNN_4Class")
    return model


model = build_model()
model.summary()

with open("model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
print("Model summary disimpan ke model_summary.txt")

# ============================================================
# 8. COMPILE & TRAIN
# ============================================================

focal_loss   = FocalLoss(gamma=2.0, alpha=0.25)
overfit_cb   = OverfittingMonitor(acc_gap_threshold=0.10)
early_stop_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1,
)
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1,
)

adam = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss=focal_loss, metrics=["accuracy"])

history = model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[overfit_cb, early_stop_cb, reduce_lr_cb],
    verbose=1,
)

# ============================================================
# 9. OVERFITTING ANALYSIS (Line Chart)
# ============================================================


def plot_overfitting_analysis(history):
    acc           = history.history["accuracy"]
    val_acc       = history.history["val_accuracy"]
    loss          = history.history["loss"]
    val_loss      = history.history["val_loss"]
    epochs        = range(1, len(acc) + 1)
    acc_gap       = [a - v for a, v in zip(acc, val_acc)]
    loss_gap_ratio = [vl / tl if tl > 0 else 1 for tl, vl in zip(loss, val_loss)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Overfitting Analysis Dashboard", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    ax1.plot(epochs, acc,     "b-o", markersize=3, label="Train Accuracy")
    ax1.plot(epochs, val_acc, "r-o", markersize=3, label="Val Accuracy")
    ax1.fill_between(epochs, acc, val_acc, alpha=0.15, color="orange", label="Gap Area")
    ax1.set_title("Accuracy: Train vs Validation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.4)

    ax2 = axes[0, 1]
    ax2.plot(epochs, loss,     "b-o", markersize=3, label="Train Loss")
    ax2.plot(epochs, val_loss, "r-o", markersize=3, label="Val Loss")
    ax2.fill_between(epochs, loss, val_loss, alpha=0.15, color="orange", label="Gap Area")
    ax2.set_title("Loss: Train vs Validation")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.4)

    ax3 = axes[1, 0]
    colors = ["red" if g > 0.10 else "green" for g in acc_gap]
    ax3.bar(epochs, acc_gap, color=colors, alpha=0.7)
    ax3.axhline(y=0.10, color="red",   linestyle="--", linewidth=1.5, label="Threshold (0.10)")
    ax3.axhline(y=0.0,  color="black", linestyle="-",  linewidth=0.8)
    ax3.set_title("Accuracy Gap per Epoch\n(Merah = Potensi Overfitting)")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Gap (Train Acc - Val Acc)")
    ax3.legend(); ax3.grid(True, alpha=0.4, axis="y")

    ax4 = axes[1, 1]
    pt_colors = ["red" if r > 1.30 else "steelblue" for r in loss_gap_ratio]
    ax4.plot(epochs, loss_gap_ratio, "k-", linewidth=1.5, alpha=0.5)
    ax4.scatter(epochs, loss_gap_ratio, c=pt_colors, s=30, zorder=5)
    ax4.axhline(y=1.30, color="red",   linestyle="--", linewidth=1.5, label="Threshold (1.30)")
    ax4.axhline(y=1.0,  color="green", linestyle="--", linewidth=1.0, label="Ideal (1.0)")
    ax4.set_title("Val/Train Loss Ratio per Epoch\n(Merah = Potensi Overfitting)")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Val Loss / Train Loss")
    ax4.legend(); ax4.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("overfitting_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    final_acc_gap    = acc_gap[-1]
    final_loss_ratio = loss_gap_ratio[-1]
    print("\n" + "=" * 50)
    print("OVERFITTING ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Final Train Accuracy : {acc[-1]:.4f}")
    print(f"Final Val Accuracy   : {val_acc[-1]:.4f}")
    print(
        f"Final Accuracy Gap   : {final_acc_gap:.4f} "
        f"{'⚠️ (> 0.10)' if final_acc_gap > 0.10 else '✅ (normal)'}"
    )
    print(
        f"Final Loss Ratio     : {final_loss_ratio:.4f} "
        f"{'⚠️ (> 1.30)' if final_loss_ratio > 1.30 else '✅ (normal)'}"
    )
    if final_acc_gap > 0.10 or final_loss_ratio > 1.30:
        print("\n⚠️  Model menunjukkan tanda-tanda OVERFITTING.")
        print("   Saran: Tambah Dropout, augmentasi data, atau kurangi kompleksitas model.")
    else:
        print("\n✅ Model dalam kondisi baik, tidak ada overfitting signifikan.")
    print("=" * 50)


plot_overfitting_analysis(history)

# ============================================================
# 10. EVALUATION
# ============================================================

print("\n=== EVALUASI PADA TEST SET ===")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

y_pred_probs = model.predict(X_test)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = np.argmax(y_test, axis=1)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Confusion Matrix — 4 Kelas")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.show()

# ============================================================
# 11. SAVE MODEL
# ============================================================

model.save("facial_expression_model.keras")
print("Model disimpan: facial_expression_model.keras")

model.export("facial_expression_savedmodel")
print("Model disimpan: facial_expression_savedmodel/")

model.save_weights("model.weights.h5")
print("Weights disimpan: model.weights.h5")

print("\n✅ Training selesai. Semua file tersimpan.")
print("   Jalankan 'python inference_camera.py' untuk real-time detection.")