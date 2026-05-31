# -*- coding: utf-8 -*-
"""
config.py — Semua konstanta dan konfigurasi FaceRead.
Diimpor oleh semua modul lain.
"""

import datetime
import os

# ── Dataset & Model ───────────────────────────────────────────
TRAIN_DIR   = "Dataset/train"
TEST_DIR    = "Dataset/test"
IMG_SIZE    = 48
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad']
NUM_CLASSES = len(CLASS_NAMES)
text_to_label = {v: k for k, v in enumerate(CLASS_NAMES)}

# ── Paths ─────────────────────────────────────────────────────
MODEL_SAVE_DIR  = "saved_models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "fr_expression_cnn.keras")
LOG_DIR         = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs("logs/fit", exist_ok=True)

# ── UI Maps ───────────────────────────────────────────────────
EMOJI_MAP = {'angry': '😠', 'happy': '😄', 'neutral': '😐', 'sad': '😢'}
COLOR_MAP = {'angry': '#ef4444', 'happy': '#f59e0b', 'neutral': '#6b7280', 'sad': '#3b82f6'}

# ── Mood ──────────────────────────────────────────────────────
MOOD_SCALE = {'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3}
MOOD_LABEL = {0: 'Sangat Positif', 1: 'Netral', 2: 'Sedih', 3: 'Marah / Frustrasi'}

# ── Learning Path ─────────────────────────────────────────────
LEARNING_PATH_MAP = {
    0: {
        'pace': 'accelerated', 'style': 'challenge-based', 'track': 'fast-track',
        'description': 'Mood sangat bagus! Cocok untuk materi menantang.',
        'phases': [
            {'phase': 1, 'topic': 'JavaScript ES2024 Advanced Patterns',  'type': 'challenge', 'duration_days': 7,  'difficulty': 'intermediate'},
            {'phase': 2, 'topic': 'React 19 + Next.js 15 App Router',     'type': 'project',   'duration_days': 14, 'difficulty': 'intermediate'},
            {'phase': 3, 'topic': 'Node.js REST API dengan Express',       'type': 'project',   'duration_days': 10, 'difficulty': 'intermediate'},
            {'phase': 4, 'topic': 'PostgreSQL + Prisma ORM',               'type': 'guided',    'duration_days': 7,  'difficulty': 'intermediate'},
        ]
    },
    1: {
        'pace': 'standard', 'style': 'guided-learning', 'track': 'standard',
        'description': 'Kondisi stabil, lanjutkan kurikulum standar.',
        'phases': [
            {'phase': 1, 'topic': 'HTML5 & CSS3 Fundamentals',          'type': 'guided',  'duration_days': 5,  'difficulty': 'beginner'},
            {'phase': 2, 'topic': 'JavaScript Dasar — DOM & Fetch API', 'type': 'guided',  'duration_days': 10, 'difficulty': 'beginner'},
            {'phase': 3, 'topic': 'React Hooks & Component Patterns',   'type': 'project', 'duration_days': 14, 'difficulty': 'intermediate'},
            {'phase': 4, 'topic': 'REST API dengan Node.js',            'type': 'guided',  'duration_days': 10, 'difficulty': 'intermediate'},
        ]
    },
    2: {
        'pace': 'relaxed', 'style': 'micro-learning', 'track': 'foundation',
        'description': 'Sesi pendek & ringan. Fokus ke hal kecil yang bisa diselesaikan.',
        'phases': [
            {'phase': 1, 'topic': 'HTML Semantics — 15 menit per sesi', 'type': 'micro',   'duration_days': 3, 'difficulty': 'beginner'},
            {'phase': 2, 'topic': 'CSS Flexbox Visual Exercises',        'type': 'visual',  'duration_days': 4, 'difficulty': 'beginner'},
            {'phase': 3, 'topic': 'JavaScript Fundamentals Review',      'type': 'review',  'duration_days': 7, 'difficulty': 'beginner'},
            {'phase': 4, 'topic': 'Simple To-Do App Project',            'type': 'project', 'duration_days': 7, 'difficulty': 'beginner'},
        ]
    },
    3: {
        'pace': 'pause-reflect', 'style': 'problem-solving', 'track': 'standard',
        'description': 'Coba istirahat sejenak. Kalau lanjut, fokus ke debugging.',
        'phases': [
            {'phase': 1, 'topic': 'Debugging Techniques & DevTools',      'type': 'skill',       'duration_days': 3, 'difficulty': 'beginner'},
            {'phase': 2, 'topic': 'Git & Version Control Best Practices', 'type': 'guided',      'duration_days': 3, 'difficulty': 'beginner'},
            {'phase': 3, 'topic': 'Error Handling Patterns in JS',        'type': 'guided',      'duration_days': 5, 'difficulty': 'intermediate'},
            {'phase': 4, 'topic': 'Code Review Simulation',               'type': 'interactive', 'duration_days': 4, 'difficulty': 'intermediate'},
        ]
    },
}