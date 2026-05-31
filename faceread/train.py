# -*- coding: utf-8 -*-
"""
train.py — Training pipeline FaceRead.
Side Quest ✅:
  - tf.GradientTape custom training loop (Phase 1)
  - TensorBoard logging
  - Split 80/20
  - Target akurasi ≥85% via MobileNetV2 Transfer Learning
"""

import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

from config import (CLASS_NAMES, IMG_SIZE, LOG_DIR, MODEL_SAVE_DIR,
                    MODEL_SAVE_PATH, NUM_CLASSES, TEST_DIR, TRAIN_DIR,
                    text_to_label)
from model import OverfittingMonitor, build_model


# ── Load gambar dari folder ───────────────────────────────────
def load_images_from_folder(folder_path: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for cls in CLASS_NAMES:
        cls_path = os.path.join(folder_path, cls)
        if not os.path.exists(cls_path):
            print(f"[WARNING] Folder tidak ditemukan: {cls_path}")
            continue
        label = text_to_label[cls]
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            img   = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(label)
    return np.array(X, dtype="float32"), np.array(y, dtype="int32")


# ── TensorBoard logging helper ────────────────────────────────
def log_scalars(writer, epoch, train_loss, train_acc, val_loss, val_acc):
    """Side Quest: TensorBoard ✅"""
    with writer.as_default():
        tf.summary.scalar("epoch_loss",         train_loss, step=epoch)
        tf.summary.scalar("epoch_accuracy",     train_acc,  step=epoch)
        tf.summary.scalar("epoch_val_loss",     val_loss,   step=epoch)
        tf.summary.scalar("epoch_val_accuracy", val_acc,    step=epoch)
    writer.flush()


# ── Phase 1: GradientTape training loop ──────────────────────
def phase1_gradient_tape(model, X_train, y_train, X_val, y_val,
                          tb_writer, epochs=20, batch_size=64, patience=8):
    """
    Custom training loop menggunakan tf.GradientTape.
    Side Quest: tf.GradientTape ✅
    """
    optimizer     = keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn       = keras.losses.CategoricalCrossentropy()
    train_acc_met = keras.metrics.CategoricalAccuracy(name="accuracy")
    val_acc_met   = keras.metrics.CategoricalAccuracy(name="val_accuracy")
    ovf_monitor   = OverfittingMonitor(acc_gap_threshold=0.10)

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(5000).batch(batch_size).prefetch(tf.data.AUTOTUNE))
    val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))

    best_val_acc = 0.0
    patience_ctr = 0
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    print("=" * 55)
    print("  PHASE 1: Custom GradientTape Loop (head only)")
    print("=" * 55)

    for epoch in range(epochs):
        train_acc_met.reset_state()
        val_acc_met.reset_state()
        epoch_loss = 0.0
        n_batches  = 0

        for x_batch, y_batch in train_ds:
            with tf.GradientTape() as tape:
                y_pred     = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, y_pred)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_acc_met.update_state(y_batch, y_pred)
            epoch_loss += loss_value.numpy()
            n_batches  += 1

        train_loss = epoch_loss / n_batches
        train_acc  = train_acc_met.result().numpy()

        val_loss_total = 0.0
        val_batches    = 0
        for x_val_b, y_val_b in val_ds:
            y_val_pred = model(x_val_b, training=False)
            val_loss_total += loss_fn(y_val_b, y_val_pred).numpy()
            val_acc_met.update_state(y_val_b, y_val_pred)
            val_batches += 1
        val_loss = val_loss_total / val_batches
        val_acc  = val_acc_met.result().numpy()

        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs} — "
              f"loss: {train_loss:.4f} — acc: {train_acc:.4f} — "
              f"val_loss: {val_loss:.4f} — val_acc: {val_acc:.4f}")

        log_scalars(tb_writer, epoch, train_loss, train_acc, val_loss, val_acc)

        ovf_monitor.on_epoch_end(epoch, logs={
            'accuracy': train_acc, 'val_accuracy': val_acc,
            'loss': train_loss,    'val_loss': val_loss
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights("best_p1.weights.h5")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n[EarlyStop] Phase 1 berhenti di epoch {epoch+1}")
                break

    model.load_weights("best_p1.weights.h5")
    ovf_monitor.on_train_end()
    return history


# ── Phase 2: Fine-tune dengan keras.fit ──────────────────────
def phase2_finetune(model, base_model, X_train, y_train, X_val, y_val):
    """Fine-tune seluruh layer dengan LR kecil."""
    base_model.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n" + "=" * 55)
    print("  PHASE 2: Fine-tuning semua layer (keras.fit)")
    print("=" * 55)

    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[
            OverfittingMonitor(acc_gap_threshold=0.10),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=15,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=5, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.TensorBoard(   # Side Quest: TensorBoard ✅
                log_dir=LOG_DIR + "_phase2",
                histogram_freq=1,
                write_graph=True,
                update_freq="epoch"
            ),
        ],
        verbose=1,
    )
    return history


# ── Plot training history ─────────────────────────────────────
def plot_history(h1: dict, h2, p1_end: int):
    def merge(key):
        return h1.get(key, []) + h2.history.get(key, [])

    acc      = merge("accuracy")
    val_acc  = merge("val_accuracy")
    loss_h   = merge("loss")
    val_loss = merge("val_loss")
    ep       = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History (Phase 1 GradientTape + Phase 2 Fine-tune)",
                 fontsize=13, fontweight="bold")

    axes[0].plot(ep, acc,     "b-", label="Train Acc")
    axes[0].plot(ep, val_acc, "r-", label="Val Acc")
    axes[0].axvline(x=p1_end, color="gray",  linestyle="--", label="Fine-tune start")
    axes[0].axhline(y=0.85,   color="green", linestyle=":",  linewidth=1.5, label="Target 85%")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(ep, loss_h,  "b-", label="Train Loss")
    axes[1].plot(ep, val_loss,"r-", label="Val Loss")
    axes[1].axvline(x=p1_end, color="gray", linestyle="--", label="Fine-tune start")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📈 Plot disimpan: training_history.png")


# ── Main training entry point ─────────────────────────────────
def run_training():
    # Load & split 80/20 ──────────────────────────────────────
    print("Loading training data...")
    X_train_raw, y_train_raw = load_images_from_folder(TRAIN_DIR)
    print(f"  Train raw: {X_train_raw.shape}")

    print("Loading test data...")
    X_test_raw, y_test_raw = load_images_from_folder(TEST_DIR)
    print(f"  Test raw : {X_test_raw.shape}")

    X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
    X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"  Split 80/20 — Train: {X_tr_raw.shape}, Test: {X_te_raw.shape}")

    # Oversampling ────────────────────────────────────────────
    ros          = RandomOverSampler(sampling_strategy="auto", random_state=42)
    X_res, y_res = ros.fit_resample(
        X_tr_raw.reshape(len(X_tr_raw), -1), y_tr_raw
    )
    print(f"  After oversampling: {X_res.shape}")

    # Preprocessing ───────────────────────────────────────────
    X_train_full = preprocess_input(
        X_res.reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    )
    X_test = preprocess_input(
        X_te_raw.reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    )
    X_train, X_val, y_tr_lbl, y_val_lbl = train_test_split(
        X_train_full, y_res, test_size=0.1, random_state=45, stratify=y_res
    )
    y_train = to_categorical(y_tr_lbl,  NUM_CLASSES)
    y_val   = to_categorical(y_val_lbl, NUM_CLASSES)
    y_test  = to_categorical(y_te_raw,  NUM_CLASSES)
    print(f"  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}\n")

    # Build model ─────────────────────────────────────────────
    model, base_model = build_model(freeze_base=True)
    model.summary()
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    # TensorBoard writer ──────────────────────────────────────
    tb_writer = tf.summary.create_file_writer(LOG_DIR)
    print(f"📊 TensorBoard logs → {LOG_DIR}")
    print(f"   Jalankan: tensorboard --logdir logs/fit\n")

    # Phase 1: GradientTape ───────────────────────────────────
    h1 = phase1_gradient_tape(model, X_train, y_train, X_val, y_val, tb_writer)
    p1_end = len(h1['accuracy'])

    # Phase 2: Fine-tune ──────────────────────────────────────
    h2 = phase2_finetune(model, base_model, X_train, y_train, X_val, y_val)

    # Plot ────────────────────────────────────────────────────
    plot_history(h1, h2, p1_end)

    # Evaluasi ────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    mae    = float(np.mean(np.abs(y_pred - y_true)))

    print(f"\n{'='*55}")
    print(f"  📊 HASIL EVALUASI")
    print(f"{'='*55}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%  {'✅' if test_acc >= 0.85 else '❌ (target ≥85%)'}")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  MAE           : {mae:.4f}  {'✅' if mae <= 0.02 else '⚠️  (target ≤0.02)'}")
    print(f"{'='*55}\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Confusion matrix ────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — Acc: {test_acc*100:.2f}%  MAE: {mae:.4f}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Simpan model — Main Quest ✅ ────────────────────────────
    model.save(MODEL_SAVE_PATH)
    model.export(os.path.join(MODEL_SAVE_DIR, "saved_model"))
    model.save_weights("model.weights.h5")

    print(f"✅ Model .keras  : {MODEL_SAVE_PATH}")
    print(f"✅ SavedModel    : {MODEL_SAVE_DIR}/saved_model")
    print(f"📊 TensorBoard  : tensorboard --logdir logs/fit")
    print(f"\n   Jalankan: python main.py --app")