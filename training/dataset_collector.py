# -* BUAT NAMBAH DATA DUMMY*-
"""
dataset_collector.py — Kumpulkan dataset ekspresi wajah via kamera
═══════════════════════════════════════════════════════════════════
Fitur:
  - Deteksi wajah otomatis (Haar Cascade)
  - Simpan gambar 48×48 grayscale ke folder dataset
  - Terorganisir per ekspresi: train/ dan test/ (80:20)
  - Preview real-time di layar
  - Keyboard shortcut per ekspresi

Kontrol:
  [A] = angry     [H] = happy
  [S] = sad       [N] = neutral
  [SPACE] = capture manual
  [Q] = quit

Jalankan:
  python dataset_collector.py

pip install opencv-python numpy
"""

import cv2
import numpy as np
import os
import time
import random
from pathlib import Path
from datetime import datetime

# ─── Konfigurasi ─────────────────────────────────────────────
DATASET_ROOT = Path("Dataset")
EXPRESSIONS  = ["angry", "happy", "sad", "neutral"]
IMG_SIZE     = 48
TRAIN_RATIO  = 0.8   # 80% train, 20% test
COUNTDOWN    = 3     # detik countdown sebelum capture otomatis
AUTO_CAPTURE = True  # True = capture otomatis saat wajah terdeteksi & tombol ditekan

# Mapping keyboard → ekspresi
KEY_MAP = {
    ord('a'): 'angry',
    ord('A'): 'angry',
    ord('h'): 'happy',
    ord('H'): 'happy',
    ord('s'): 'sad',
    ord('S'): 'sad',
    ord('n'): 'neutral',
    ord('N'): 'neutral',
}

# Warna per ekspresi (BGR)
COLOR_MAP = {
    'angry'  : (0, 0, 220),
    'happy'  : (0, 200, 255),
    'sad'    : (220, 100, 0),
    'neutral': (180, 180, 180),
}

# ─── Setup Folder ────────────────────────────────────────────
def setup_folders():
    for split in ['train', 'test']:
        for expr in EXPRESSIONS:
            path = DATASET_ROOT / split / expr
            path.mkdir(parents=True, exist_ok=True)
    print("✅ Folder dataset siap:")
    for split in ['train', 'test']:
        for expr in EXPRESSIONS:
            count = len(list((DATASET_ROOT / split / expr).glob('*.png')))
            print(f"   {split}/{expr}: {count} gambar")

def count_images():
    counts = {}
    for expr in EXPRESSIONS:
        train_n = len(list((DATASET_ROOT / 'train' / expr).glob('*.png')))
        test_n  = len(list((DATASET_ROOT / 'test'  / expr).glob('*.png')))
        counts[expr] = {'train': train_n, 'test': test_n, 'total': train_n + test_n}
    return counts

def save_face(face_img, expression):
    """Simpan gambar wajah ke folder train atau test (80:20 split otomatis)."""
    split     = 'train' if random.random() < TRAIN_RATIO else 'test'
    folder    = DATASET_ROOT / split / expression
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename  = f"{expression}_{timestamp}.png"
    filepath  = folder / filename

    # Resize ke 48×48 dan grayscale
    face_gray    = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))

    cv2.imwrite(str(filepath), face_resized)
    return filepath, split

# ─── Main App ────────────────────────────────────────────────
def draw_ui(frame, active_expr, counts, status_msg, status_color, face_box):
    h, w = frame.shape[:2]

    # Overlay gelap di bagian bawah
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-160), (w, h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Overlay di atas
    cv2.rectangle(overlay, (0, 0), (w, 50), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "DATASET COLLECTOR", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Kotak wajah
    if face_box is not None:
        x, y, fw, fh = face_box
        col = COLOR_MAP.get(active_expr, (255,255,255)) if active_expr else (255,255,255)
        cv2.rectangle(frame, (x,y), (x+fw,y+fh), col, 2)
        # Sudut dekoratif
        sz = 15
        cv2.line(frame, (x,y), (x+sz,y), col, 3)
        cv2.line(frame, (x,y), (x,y+sz), col, 3)
        cv2.line(frame, (x+fw,y), (x+fw-sz,y), col, 3)
        cv2.line(frame, (x+fw,y), (x+fw,y+sz), col, 3)
        cv2.line(frame, (x,y+fh), (x+sz,y+fh), col, 3)
        cv2.line(frame, (x,y+fh), (x,y+fh-sz), col, 3)
        cv2.line(frame, (x+fw,y+fh), (x+fw-sz,y+fh), col, 3)
        cv2.line(frame, (x+fw,y+fh), (x+fw,y+fh-sz), col, 3)

    # Panel bawah — kontrol
    y_base = h - 150
    controls = [
        ("[A] Angry",   'angry'),
        ("[H] Happy",   'happy'),
        ("[S] Sad",     'sad'),
        ("[N] Neutral", 'neutral'),
    ]
    col_w = w // len(controls)
    for i, (label, expr) in enumerate(controls):
        x_pos  = i * col_w + 10
        col    = COLOR_MAP[expr]
        active = (expr == active_expr)
        if active:
            cv2.rectangle(frame, (i*col_w, y_base-5), ((i+1)*col_w-2, y_base+35), col, -1)
            cv2.putText(frame, label, (x_pos, y_base+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
        else:
            cv2.putText(frame, label, (x_pos, y_base+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

        # Jumlah gambar
        total = counts.get(expr, {}).get('total', 0)
        cv2.putText(frame, f"{total} foto", (x_pos, y_base+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)

    # Status message
    cv2.putText(frame, status_msg, (10, h-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Info bawah
    cv2.putText(frame, "[SPACE] Capture  |  [Q] Quit", (10, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)

    return frame

def main():
    setup_folders()
    print("\n📷 Kamera aktif. Kontrol:")
    print("  [A]=angry  [H]=happy  [S]=sad  [N]=neutral")
    print("  [SPACE]=capture  [Q]=quit\n")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    active_expr  = None
    status_msg   = "Pilih ekspresi: [A]ngry [H]appy [S]ad [N]eutral"
    status_color = (180, 180, 180)
    last_capture = 0
    cooldown     = 0.8   # detik antar capture

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        face_box  = None
        face_crop = None
        if len(faces) > 0:
            # Ambil wajah terbesar
            x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            face_box  = (x, y, fw, fh)
            face_crop = frame[y:y+fh, x:x+fw]

        counts = count_images()
        frame  = draw_ui(frame, active_expr, counts, status_msg, status_color, face_box)

        cv2.imshow("Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q') or key == ord('Q') or key == 27:
            break

        # Pilih ekspresi
        if key in KEY_MAP:
            active_expr  = KEY_MAP[key]
            status_msg   = f"Ekspresi: {active_expr.upper()} — arahkan wajah lalu tekan SPACE"
            status_color = COLOR_MAP[active_expr]
            print(f"[MODE] {active_expr.upper()}")

        # Capture
        if key == ord(' ') and active_expr is not None:
            now = time.time()
            if now - last_capture < cooldown:
                continue
            if face_crop is None:
                status_msg   = "⚠ Wajah tidak terdeteksi! Pastikan wajah terlihat."
                status_color = (0, 100, 255)
            else:
                filepath, split = save_face(face_crop, active_expr)
                last_capture    = now
                total           = counts[active_expr]['total'] + 1
                status_msg      = f"✅ Tersimpan! {active_expr} → {split}/ (total: {total})"
                status_color    = COLOR_MAP[active_expr]
                print(f"[SAVE] {filepath}")

                # Flash efek
                flash = frame.copy()
                cv2.rectangle(flash, (0,0), (frame.shape[1], frame.shape[0]),
                              COLOR_MAP[active_expr], 8)
                cv2.imshow("Dataset Collector", flash)
                cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()

    # Ringkasan akhir
    print("\n=== RINGKASAN DATASET ===")
    counts = count_images()
    total_all = 0
    for expr in EXPRESSIONS:
        c = counts[expr]
        print(f"  {expr:10s} → train: {c['train']:4d} | test: {c['test']:4d} | total: {c['total']:4d}")
        total_all += c['total']
    print(f"  {'TOTAL':10s} → {total_all} gambar baru ditambahkan")
    print("=========================\n")
    print("✅ Selesai! Dataset siap digunakan untuk training.")

if __name__ == "__main__":
    main()
