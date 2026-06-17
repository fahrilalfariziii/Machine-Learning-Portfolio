# 🛍️ Purchase Conversion Prediction Platform

[[link dataset](https://www.kaggle.com/datasets/anujsaha0123456789/e-commerce-product-intelligence-dataset)]

## 📌 Ringkasan Proyek

**Purchase Conversion Prediction Platform** adalah proyek portofolio machine learning _end-to-end_ yang membangun sistem prediksi probabilitas konversi pengguna e-commerce. Proyek ini mengintegrasikan:

- model CatBoost untuk scoring peluang pembelian,
- backend FastAPI untuk melayani prediksi sebagai API,
- frontend Streamlit untuk demonstrasi antarmuka pengguna interaktif.

---

## 🎯 Tujuan Bisnis

Masalah utama yang diselesaikan adalah memperkirakan apakah interaksi pengguna (misalnya tampilan, klik, atau tambah ke keranjang) akan menghasilkan pembelian.

Manfaat solusi ini:

- Mengurangi anggaran pemasaran yang terbuang pada pengguna non-potensial.
- Mengarahkan promosi dan diskon ke segmen pengguna dengan probabilitas pembelian tinggi.
- Mendukung keputusan berbasis data untuk kampanye retensi dan akuisisi.

---

## 🧠 Arsitektur Sistem

Proyek ini dirancang dengan arsitektur terpisah antara frontend dan backend:

```text
[ Browser ] -- Streamlit UI (8501) --> FastAPI Backend (8000) --> CatBoost Model
```

- `app/streamlit_app.py` = antarmuka input interaksi pengguna dan visualisasi hasil.
- `app/main.py` = backend FastAPI yang menerima request JSON dan mengembalikan probabilitas pembelian.
- `models/purchase_conversion_model.pkl` = model CatBoost yang dilatih.

---

## 🧩 Fitur Utama

- Prediksi probabilitas konversi interaksi pengguna
- Klasifikasi keputusan bisnis: `POTENSI TINGGI` atau `POTENSI RENDAH`
- Input interaksi, profil pengguna, dan atribut produk
- Struktur API RESTful untuk pemisahan frontend/backend
- Dukungan deployment lokal dan Docker

---

## 📂 Struktur Proyek

```text
purchase-conversion-platform/
├── app/
│   ├── main.py          # Server Backend FastAPI & Logika Inferensi
│   └── streamlit_app.py # Dashboard Frontend Streamlit (User Interface)
├── models/
│   └── purchase_conversion_catboost.pkl # Artifact Model CatBoost
├── src/
│   └── purchase-conversion-prediction-platform.ipynb       # Skrip Pipeline Training Utama (Kaggle/Local)
├── Dockerfile           # Konfigurasi image Python & System Dependencies
├── docker-compose.yml   # Orkestrasi Container Backend & Frontend
└── requirements.txt     # Daftar Library Python Teruji
```

---

## 📦 Instalasi & Menjalankan Lokal

1. Buat virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan backend FastAPI:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Jalankan frontend Streamlit di terminal lain:
   ```bash
   streamlit run app/streamlit_app.py --server.port 8501
   ```
5. Buka browser ke:
   - `http://127.0.0.1:8501` untuk UI Streamlit
   - `http://127.0.0.1:8000/docs` untuk dokumentasi API OpenAPI

---

## 🐳 Jalankan dengan Docker

```bash
docker-compose up --build
```

Kemudian akses:

- `http://localhost:8501` untuk Streamlit
- `http://localhost:8000/docs` untuk API FastAPI

---

## 📌 Target Variabel

Model memprediksi label `purchase_label`:

| Nilai | Keterangan                       |
| ----- | -------------------------------- |
| 1     | Interaksi menghasilkan pembelian |
| 0     | Tidak menghasilkan pembelian     |

---

## 🧪 Catatan

- Model disimpan di `models/purchase_conversion_model.pkl`.
- Frontend Streamlit mengirim data ke endpoint FastAPI menggunakan `FASTAPI_URL`.
- File `purchase-conversion-prediction-platform.ipynb` adalah notebook analisis dan eksperimen.

---

## 📌 Teknologi

- Python
- FastAPI
- Streamlit
- CatBoost
- pandas, numpy
- Docker / docker-compose
