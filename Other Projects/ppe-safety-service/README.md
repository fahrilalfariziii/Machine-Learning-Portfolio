# 🛡️ PPE Safety Detection Service

Sistem deteksi dan monitoring kepatuhan Alat Pelindung Diri (APD) berbasis kecerdasan buatan menggunakan YOLOv10. Aplikasi ini dirancang untuk industri manufaktur, konstruksi, dan area kerja berisiko tinggi untuk memastikan keselamatan kerja karyawan secara real-time.

## 📋 Fitur Utama

- **Deteksi Real-Time**: Mendeteksi helm keselamatan dan rompi pelindung secara akurat
- **API FastAPI**: Backend service yang scalable untuk integrasi dengan sistem lain
- **Dashboard Streamlit**: Interface user-friendly untuk monitoring dan analisis
- **Model YOLOv10**: Menggunakan arsitektur YOLO terdepan untuk akurasi tinggi
- **Docker Support**: Mudah di-deploy di berbagai environment
- **Confidence Threshold Adjustable**: Kontrol sensitivitas deteksi sesuai kebutuhan

## 🏗️ Struktur Proyek

```
ppe-safety-service/
├── README.md                    # File dokumentasi ini
├── dockerfile                   # Konfigurasi Docker
├── requirements.txt             # Dependencies Python
├── personal-protective-equipment-detection-yolov10.ipynb  # Notebook training
├── app/
│   ├── main.py                  # Backend FastAPI service
│   └── ui.py                    # Frontend Streamlit dashboard
└── weights/
    └── safety_best_model.pt     # Pre-trained YOLOv10 model
```

## 📦 Persyaratan Sistem

- Python 3.8+
- pip atau conda
- CUDA 11.0+ (opsional, untuk GPU acceleration)
- Docker (opsional, untuk containerization)

## 🚀 Instalasi & Setup

### Opsi 1: Local Installation

1. **Clone atau download proyek**

   ```bash
   cd ppe-safety-service
   ```

2. **Buat virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Opsi 2: Docker Installation

1. **Build Docker image**

   ```bash
   docker build -t ppe-safety-service .
   ```

2. **Run Docker container**
   ```bash
   docker run -p 8000:8000 -p 8501:8501 ppe-safety-service
   ```

## ▶️ Cara Menjalankan

### Backend FastAPI Server

Jalankan service API detection:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server akan berjalan di: `http://localhost:8000`

API Documentation tersedia di: `http://localhost:8000/docs`

### Frontend Streamlit Dashboard

Pada terminal baru, jalankan dashboard:

```bash
streamlit run app/ui.py
```
Dashboard akan terbuka di: `http://localhost:8501`

## 🎨 Dashboard Streamlit

Fitur dashboard:

- **Upload Gambar**: Unggah foto area kerja
- **Adjustment Confidence**: Slider untuk mengatur threshold deteksi (0.1 - 1.0)
- **Visualisasi Hasil**: Tampilan side-by-side antara gambar original dan hasil deteksi
- **Statistik Deteksi**: Jumlah objek yang terdeteksi dan confidence scores

## 🔧 Dependencies

| Package                | Versi    | Fungsi               |
| ---------------------- | -------- | -------------------- |
| fastapi                | 0.110.0  | Backend framework    |
| uvicorn                | 0.28.0   | ASGI server          |
| ultralytics            | 8.1.34   | YOLOv10 framework    |
| opencv-python-headless | 4.9.0.80 | Image processing     |
| numpy                  | 1.26.4   | Numerical computing  |
| streamlit              | 1.32.0   | Frontend framework   |
| requests               | 2.31.0   | HTTP requests        |
| python-multipart       | 0.0.9    | File upload handling |

## 📚 Jupyter Notebook

File `personal-protective-equipment-detection-yolov10.ipynb` berisi:

- Exploratory Data Analysis (EDA)
- Model training dan fine-tuning
- Evaluation metrics
- Visualization hasil deteksi

Jalankan notebook untuk training atau eksperimen lebih lanjut:

```bash
jupyter notebook personal-protective-equipment-detection-yolov10.ipynb
```