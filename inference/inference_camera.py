# -*- coding: utf-8 -*-
"""
inference_camera.py
Real-Time Facial Expression Recognition — Simple Camera Interface
Jalankan dengan: python inference/inference_camera.py
"""

import os
import time
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras as keras_core
import threading

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    print("[WARN] pip install requests — diperlukan untuk POST ke API")

# ============================================================
# CUSTOM OBJECTS — harus identik dengan cek.py agar load_model tidak error
# Catatan: ChannelAttention sudah dihapus dari cek.py, TIDAK didaftarkan lagi.
#          FocalLoss tetap didaftarkan untuk backward compat model lama.
# ============================================================

@keras_core.saving.register_keras_serializable(package="FER")
class CustomCrossEntropy(keras.losses.Loss):
    def __init__(self, smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, label_smoothing=self.smoothing
        )

    def get_config(self):
        config = super().get_config()
        config.update({"smoothing": self.smoothing})
        return config


@keras_core.saving.register_keras_serializable(package="FER")
class FocalLoss(keras.losses.Loss):
    """Backward compat — model lama mungkin masih pakai ini."""
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = self.alpha * tf.pow(1.0 - pt, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * cross_entropy, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/saved_models/facial_expression_model.keras"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

IMG_SIZE    = 48     # ukuran crop wajah dari cascade
TARGET_SIZE = 224    # input ResNet50

EMOTIONS = ["angry", "happy", "sad", "neutral"]

EMOTION_COLORS = {
    "angry":   (220, 50,  50),
    "happy":   (50,  200, 50),
    "sad":     (50,  100, 220),
    "neutral": (120, 120, 120),
}
EMOTION_EMOJI = {
    "angry":   "😠",
    "happy":   "😄",
    "sad":     "😢",
    "neutral": "😐",
}

CONF_THRESHOLD = 0.25
API_URL        = "http://localhost:8000/checkin"
CHECKIN_WINDOW = 4.0   # detik pertama untuk voting mood

# ============================================================
# LOAD MODEL & CASCADE
# ============================================================

def load_resources():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] File model tidak ditemukan: {MODEL_PATH}")
        raise SystemExit(1)

    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print(f"[OK] Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        raise SystemExit(1)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print(f"[ERROR] Haarcascade tidak ditemukan: {CASCADE_PATH}")
        raise SystemExit(1)
    print("[OK] Cascade loaded.")
    return model, cascade

# ============================================================
# INFERENCE HELPER
# ============================================================

def preprocess_face(gray_img, x, y, w, h):
    """
    Preprocessing wajah — harus sama persis dengan cek.py:
      1. Crop ROI dari grayscale frame
      2. Resize ke 48x48
      3. Grayscale → RGB (3 channel)
      4. Resize ke 224x224
      5. preprocess_input ResNet50
      6. Tambah batch dim → (1, 224, 224, 3)
    """
    roi = gray_img[y: y + h, x: x + w]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    roi_resized = cv2.resize(roi_rgb, (TARGET_SIZE, TARGET_SIZE))
    roi_float = roi_resized.astype("float32")
    roi_preprocessed = tf.keras.applications.resnet50.preprocess_input(roi_float)
    return np.expand_dims(roi_preprocessed, axis=0)


def predict_emotion(model, face_input):
    """face_input: array (1, 224, 224, 3)."""
    preds = model.predict(face_input, verbose=0)[0]
    idx   = int(np.argmax(preds))
    conf  = float(preds[idx])
    label = EMOTIONS[idx] if conf >= CONF_THRESHOLD else "uncertain"
    return label, conf, preds

# ============================================================
# MAIN GUI APPLICATION
# ============================================================

class FERApp:
    def __init__(self, root, model, cascade):
        self.root    = root
        self.model   = model
        self.cascade = cascade
        self.cap     = None
        self.running = False

        self._cam_start    = 0.0
        self._votes        = {}
        self._best_preds   = {}
        self._checkin_done = False

        self._setup_ui()
        self._start_camera()

    # ── UI SETUP ──────────────────────────────────────────────
    def _setup_ui(self):
        self.root.title("Facial Expression Recognition — Real Time")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")

        header = tk.Frame(self.root, bg="#16213e", pady=8)
        header.pack(fill="x")
        tk.Label(header, text="🎭  Facial Expression Recognition",
                 font=("Segoe UI", 16, "bold"), fg="#e94560", bg="#16213e").pack()
        tk.Label(header, text="Real-Time ResNet50 Detection",
                 font=("Segoe UI", 10), fg="#a8a8b3", bg="#16213e").pack()

        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(padx=10, pady=10)

        cam_frame = tk.Frame(main, bg="#0f3460", bd=2, relief="ridge")
        cam_frame.grid(row=0, column=0, padx=(0, 10))
        self.canvas = tk.Canvas(cam_frame, width=640, height=480,
                                bg="#000", highlightthickness=0)
        self.canvas.pack()

        info = tk.Frame(main, bg="#1a1a2e", width=240)
        info.grid(row=0, column=1, sticky="nsew")
        info.pack_propagate(False)

        tk.Label(info, text="Detected Emotion", font=("Segoe UI", 11, "bold"),
                 fg="#a8a8b3", bg="#1a1a2e").pack(pady=(0, 4))

        self.emotion_label = tk.Label(info, text="—", font=("Segoe UI", 28, "bold"),
                                      fg="#e94560", bg="#1a1a2e")
        self.emotion_label.pack()

        self.emoji_label = tk.Label(info, text="", font=("Segoe UI", 44), bg="#1a1a2e")
        self.emoji_label.pack(pady=4)

        self.conf_label = tk.Label(info, text="Confidence: —",
                                   font=("Segoe UI", 10), fg="#a8a8b3", bg="#1a1a2e")
        self.conf_label.pack()

        tk.Label(info, text="", bg="#1a1a2e").pack(pady=2)
        self.conf_bar = ttk.Progressbar(info, length=200,
                                        mode="determinate", maximum=100)
        self.conf_bar.pack()

        tk.Frame(info, bg="#0f3460", height=2).pack(fill="x", pady=10)
        tk.Label(info, text="All Probabilities", font=("Segoe UI", 10, "bold"),
                 fg="#a8a8b3", bg="#1a1a2e").pack()

        self.prob_bars = {}
        self.prob_vals = {}
        for emo in EMOTIONS:
            row = tk.Frame(info, bg="#1a1a2e")
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=f"{EMOTION_EMOJI.get(emo, '')} {emo:<9}",
                     font=("Courier", 9), fg="#c0c0d0", bg="#1a1a2e",
                     width=13, anchor="w").pack(side="left")
            bar = ttk.Progressbar(row, length=110, mode="determinate", maximum=100)
            bar.pack(side="left", padx=3)
            val_lbl = tk.Label(row, text="0%", font=("Courier", 9),
                               fg="#a8a8b3", bg="#1a1a2e", width=5)
            val_lbl.pack(side="left")
            self.prob_bars[emo] = bar
            self.prob_vals[emo] = val_lbl

        tk.Frame(info, bg="#0f3460", height=2).pack(fill="x", pady=10)
        self.face_count_label = tk.Label(info, text="Faces: 0",
                                         font=("Segoe UI", 10), fg="#a8a8b3", bg="#1a1a2e")
        self.face_count_label.pack()
        self.fps_label = tk.Label(info, text="FPS: —",
                                  font=("Segoe UI", 9), fg="#555577", bg="#1a1a2e")
        self.fps_label.pack()

        tk.Frame(info, bg="#0f3460", height=2).pack(fill="x", pady=6)
        tk.Label(info, text="Daily Check-in", font=("Segoe UI", 10, "bold"),
                 fg="#a8a8b3", bg="#1a1a2e").pack()
        self.checkin_status = tk.Label(
            info, text="⏳ Mengumpulkan mood (4 detik)...",
            font=("Segoe UI", 9), fg="#f39c12", bg="#1a1a2e",
            wraplength=220, justify="center")
        self.checkin_status.pack(pady=4)

        # Tampilkan AI suggestion setelah check-in
        tk.Frame(info, bg="#0f3460", height=2).pack(fill="x", pady=4)
        tk.Label(info, text="AI Suggestion", font=("Segoe UI", 10, "bold"),
                 fg="#a8a8b3", bg="#1a1a2e").pack()
        self.ai_label = tk.Label(
            info, text="—",
            font=("Segoe UI", 9), fg="#a8d8ea", bg="#1a1a2e",
            wraplength=220, justify="center")
        self.ai_label.pack(pady=4)

        ctrl = tk.Frame(self.root, bg="#1a1a2e", pady=8)
        ctrl.pack(fill="x", padx=10)
        btn_cfg = {"font": ("Segoe UI", 10, "bold"), "bd": 0, "relief": "flat",
                   "cursor": "hand2", "padx": 16, "pady": 6}

        self.btn_toggle = tk.Button(ctrl, text="⏸  Pause", bg="#e94560", fg="white",
                                    command=self._toggle_detection, **btn_cfg)
        self.btn_toggle.pack(side="left", padx=4)
        tk.Button(ctrl, text="📸  Screenshot", bg="#0f3460", fg="white",
                  command=self._screenshot, **btn_cfg).pack(side="left", padx=4)
        tk.Button(ctrl, text="❌  Exit", bg="#333355", fg="white",
                  command=self._on_close, **btn_cfg).pack(side="right", padx=4)

        self.status_var = tk.StringVar(value="Initializing camera...")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Segoe UI", 8), fg="#555577", bg="#1a1a2e", anchor="w"
                 ).pack(fill="x", padx=10, pady=(0, 6))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── CAMERA ────────────────────────────────────────────────
    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Kamera tidak dapat dibuka!")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running    = True
        self._cam_start = time.time()
        self.status_var.set("Camera active — tekan Exit atau tutup jendela untuk keluar")
        self._update_frame()

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        t0 = time.time()
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.32, minNeighbors=5)

        dominant_label = "—"
        dominant_conf  = 0.0
        dominant_preds = None

        for x, y, w, h in faces:
            face_input = preprocess_face(gray, x, y, w, h)
            label, conf, preds = predict_emotion(self.model, face_input)

            color_rgb = EMOTION_COLORS.get(label, (200, 200, 200))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, thickness=3)

            label_text = f"{label} ({conf * 100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color_bgr, -1)
            cv2.putText(frame, label_text, (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if dominant_conf < conf:
                dominant_label = label
                dominant_conf  = conf
                dominant_preds = preds

        # Check-in 4 detik pertama
        elapsed = time.time() - self._cam_start
        if not self._checkin_done:
            if elapsed <= CHECKIN_WINDOW:
                if dominant_label != "—" and dominant_preds is not None:
                    self._votes[dominant_label] = self._votes.get(dominant_label, 0) + 1
                    self._best_preds[dominant_label] = dominant_preds
                sisa = max(0, CHECKIN_WINDOW - elapsed)
                self.checkin_status.config(
                    text=f"⏳ Mengumpulkan mood... ({sisa:.1f}s)", fg="#f39c12")
            else:
                self._checkin_done = True
                self._post_checkin_background()

        self.face_count_label.config(text=f"Faces: {len(faces)}")
        fps = 1.0 / max(time.time() - t0, 1e-6)
        self.fps_label.config(text=f"FPS: {fps:.1f}")

        if dominant_preds is not None:
            self._update_info_panel(dominant_label, dominant_conf, dominant_preds)
        else:
            self._reset_info_panel()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tk    = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.canvas.img_tk = img_tk
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)

        self.root.after(15, self._update_frame)

    # ── POST CHECK-IN ─────────────────────────────────────────
    def _post_checkin_background(self):
        if not self._votes:
            self.checkin_status.config(
                text="❌ Tidak ada wajah terdeteksi\ndi 4 detik pertama", fg="#e74c3c")
            return

        best_mood  = max(self._votes, key=self._votes.get)
        best_preds = self._best_preds.get(best_mood)
        total      = sum(self._votes.values())
        conf_pct   = round(self._votes[best_mood] / total * 100, 1)

        self.checkin_status.config(text="🔄 Mengirim mood ke server...", fg="#f39c12")

        def _do_post():
            if not REQUESTS_OK:
                self.root.after(0, lambda: self.checkin_status.config(
                    text="⚠️ requests tidak ter-install\npip install requests",
                    fg="#e74c3c"))
                return
            try:
                payload = {
                    "user_id":        "default",
                    "mood":           best_mood,
                    "confidence":     conf_pct,
                    "emoji":          EMOTION_EMOJI.get(best_mood, ""),
                    "all_probs":      {
                        EMOTIONS[i]: round(float(best_preds[i]) * 100, 1)
                        for i in range(len(EMOTIONS))
                    } if best_preds is not None else {},
                    # ✅ FIX: default True agar AI suggestion aktif
                    "use_ai_comment": True,
                }
                resp = requests.post(API_URL, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()

                    # ✅ FIX: akses data["data"]["id"] bukan data["id"]
                    checkin_id = data["data"]["id"]
                    ai         = data["data"].get("ai_suggestion")
                    ai_msg     = ai["message"] if ai else "—"

                    txt = (
                        f"✅ Tersimpan! ID #{checkin_id}\n"
                        f"{EMOTION_EMOJI.get(best_mood, '')} {best_mood.upper()}"
                    )
                    self.root.after(0, lambda: self.checkin_status.config(
                        text=txt, fg="#2ecc71"))
                    self.root.after(0, lambda: self.ai_label.config(
                        text=ai_msg, fg="#a8d8ea"))
                    self.root.after(0, lambda: self.status_var.set(
                        f"Check-in berhasil — mood: {best_mood.upper()} | Kamera tetap jalan"))
                else:
                    self.root.after(0, lambda: self.checkin_status.config(
                        text=f"❌ Server error {resp.status_code}", fg="#e74c3c"))
            except Exception as e:
                err = str(e)[:50]
                self.root.after(0, lambda: self.checkin_status.config(
                    text=f"⚠️ API tidak tersambung\n{err}", fg="#e74c3c"))

        threading.Thread(target=_do_post, daemon=True).start()

    # ── INFO PANEL ────────────────────────────────────────────
    def _update_info_panel(self, label, conf, preds):
        color = "#%02x%02x%02x" % EMOTION_COLORS.get(label, (200, 200, 200))
        self.emotion_label.config(text=label.upper(), fg=color)
        self.emoji_label.config(text=EMOTION_EMOJI.get(label, ""))
        self.conf_label.config(text=f"Confidence: {conf * 100:.1f}%")
        self.conf_bar["value"] = conf * 100
        for i, emo in enumerate(EMOTIONS):
            p = float(preds[i]) * 100
            self.prob_bars[emo]["value"] = p
            self.prob_vals[emo].config(text=f"{p:.0f}%")

    def _reset_info_panel(self):
        self.emotion_label.config(text="—", fg="#e94560")
        self.emoji_label.config(text="")
        self.conf_label.config(text="Confidence: —")
        self.conf_bar["value"] = 0
        for emo in EMOTIONS:
            self.prob_bars[emo]["value"] = 0
            self.prob_vals[emo].config(text="0%")

    # ── CONTROLS ──────────────────────────────────────────────
    def _toggle_detection(self):
        if self.running:
            self.running = False
            self.btn_toggle.config(text="▶  Resume", bg="#28a745")
            self.status_var.set("Paused")
        else:
            self.running = True
            self.btn_toggle.config(text="⏸  Pause", bg="#e94560")
            self.status_var.set("Camera active")
            self._update_frame()

    def _screenshot(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, cv2.flip(frame, 1))
            self.status_var.set(f"Screenshot saved: {fname}")
            messagebox.showinfo("Screenshot", f"Tersimpan: {fname}")

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Loading model & cascade...")
    model, cascade = load_resources()

    root  = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Horizontal.TProgressbar",
                    troughcolor="#2a2a4a", background="#e94560", thickness=12)

    app = FERApp(root, model, cascade)
    root.mainloop()