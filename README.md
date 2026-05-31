# MindCare: Facial Expression Recognition System

MindCare adalah sistem pengenalan ekspresi wajah (Facial Expression Recognition) berbasis Deep Learning menggunakan arsitektur ResNet50, lengkap dengan REST API (FastAPI) untuk integrasi dengan antarmuka pengguna (Frontend React).

## 📁 Struktur Proyek

Proyek ini terbagi menjadi beberapa modul utama:

- **`training/`**: Berisi script untuk melatih model klasifikasi ekspresi wajah (4 kelas: angry, happy, sad, neutral).
  - `mindcare.py`: Script utama untuk *fine-tuning* model ResNet50.
  - `dataset_collector.py`: Script untuk mengumpulkan dataset wajah.
- **`inference/`**: Berisi script untuk deteksi secara *real-time* dan integrasi API.
  - `inference_camera.py`: Deteksi mood/ekspresi secara langsung menggunakan kamera pengguna.
  - `inference_api.py`: FastAPI REST API untuk menerima hasil deteksi, menyimpannya ke database SQLite (`mood_checkins.db`), dan menyediakan *endpoint* untuk frontend.
  - `ai_suggestion.py`: Modul terintegrasi AI (Anthropic) untuk memberikan saran berbasis mood yang terdeteksi.
- **`models/` & `saved_models/`**: Direktori penyimpanan model yang telah dilatih (format `.keras` maupun format TF SavedModel).
- **`Dataset/`**: Direktori penyimpanan dataset (train & test) untuk melatih model klasifikasi ekspresi wajah.

## 🚀 Fitur Utama

1. **Deteksi Emosi Real-time**: Memanfaatkan model transfer learning ResNet50 untuk memprediksi 4 emosi secara akurat.
2. **REST API Endpoint**: Server backend FastAPI yang siap dihubungkan dengan frontend (menyediakan fitur check-in harian).
3. **AI Suggestion**: Saran kustom harian berdasarkan emosi yang terdeteksi.
4. **Analisis Overfitting**: Dilengkapi dengan plotting evaluasi model (terdapat `overfitting_analysis.png` & `confusion_matrix.png`).

## 🛠️ Instalasi & Persiapan Lingkungan

Disarankan menggunakan *Virtual Environment* (venv). Pada direktori `face_recog` sudah terdapat konfigurasi virtual environment yang bisa digunakan.

1. **Aktivasi Environment (Windows):**
   ```bash
   face_recog\Scripts\activate
   ```

2. **Install Dependensi:**
   Pastikan library berikut diinstal:
   ```bash
   pip install tensorflow keras opencv-python fastapi uvicorn sqlalchemy anthropic scikit-learn imbalanced-learn matplotlib seaborn
   ```

## 💻 Cara Menjalankan

### 1. Training Model (Opsional, untuk melatih ulang)
Pastikan dataset ada di dalam folder `Dataset/train` dan `Dataset/test`.
```bash
python training/mindcare.py
```
*Akan menghasilkan model di folder `saved_models` serta grafik akurasi.*

### 2. Menjalankan Server REST API
Buka terminal baru, aktifkan environment, lalu jalankan:
```bash
uvicorn inference.inference_api:app --host 0.0.0.0 --port 8000 --reload
```
*Dokumentasi interaktif (Swagger UI) tersedia di: `http://localhost:8000/docs`*

### 3. Menjalankan Deteksi Kamera
Di terminal lainnya, jalankan detektor kamera yang akan mengirim hasil prediksi ke API:
```bash
python inference/inference_camera.py
```

---

### 🔗 Link Referensi & Sumber Daya Eksternal
*Dataset, Model, atau Asset tambahan:*
[Google Drive Folder](https://drive.google.com/drive/folders/1GTLCOxTTNUwo6qW3yalo1UN-Jw4b5_GC?usp=sharing)