---
title: IkanJi - Deteksi Kesegaran Ikan AI
emoji: 🐟
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🐟 FreshScan – Fish Freshness Detection AI

Aplikasi web berbasis AI untuk mendeteksi tingkat kesegaran ikan laut dari foto insang menggunakan **MobileNetV2 + Transfer Learning**.

## ✨ Fitur
- Upload foto insang via drag & drop atau klik
- Klasifikasi 3 kelas: Sangat Segar, Segar Biasa, Tidak Segar
- Tampilkan distribusi probabilitas tiap kelas
- Akurasi model: **99%** (F1-score weighted 0.99)
- UI premium, dark mode, responsive

## 🗂️ Struktur Proyek
```
fish-freshness-app/
├── app.py                  # Flask backend
├── requirements.txt        # Python dependencies
├── Procfile               # Deploy config (Heroku/Railway)
├── runtime.txt            # Python version
├── model_insang_ikan_pytorch.pth  # Model (taruh di sini)
└── templates/
    └── index.html         # Frontend UI
```

## 🚀 Cara Menjalankan Lokal

### 1. Salin file model
```
cp /path/ke/model_insang_ikan_pytorch.pth ./
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan server
```bash
python app.py
```

Buka browser di: **http://localhost:5000**

---

## ☁️ Deploy ke Railway (Gratis)

1. Buat akun di [railway.app](https://railway.app)
2. Buat project baru → **Deploy from GitHub**
3. Upload/push folder ini ke GitHub (pastikan file `.pth` ikut diupload)
4. Railway otomatis detect `Procfile` dan deploy

> **PENTING**: File model `.pth` (ukuran ~14MB) harus ikut di-commit ke repo karena tidak ada cloud storage yang dikonfigurasi.

---

## ☁️ Deploy ke Render (Gratis)

1. Buat akun di [render.com](https://render.com)
2. New → **Web Service** → Connect GitHub repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app --workers 1 --timeout 120`
5. Environment: Python 3.11

---

## 🧠 Arsitektur Model

| Parameter | Nilai |
|-----------|-------|
| Arsitektur | MobileNetV2 |
| Metode | Transfer Learning (Fine-tuning) |
| Input Size | 224 × 224 piksel |
| Kelas | 3 (Sangat Segar, Segar Biasa, Tidak Segar) |
| Dataset | 1.755 citra insang |
| Optimizer | Adam (lr=0.0001) |
| Epochs | 10 |
| Akurasi Validasi | **99%** |

## 👥 Penulis
- **Safaril Adam** – Teknik Informatika, Universitas Halu Oleo
- **Maya Agustin** – Teknik Informatika, Universitas Halu Oleo
