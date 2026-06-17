# Menggunakan base image Python yang ringan
FROM python:3.11-slim

# Mengatur environment variable agar Python tidak menulis file .pyc dan output langsung ke terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Menentukan direktori kerja di dalam container
WORKDIR /workspace

# Menginstal dependensi sistem yang dibutuhkan oleh CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Menyalin berkas requirements dan menginstal library Python
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode proyek ke dalam container
COPY . /workspace/

# Ekspos port untuk FastAPI (8000) dan Streamlit (8501)
EXPOSE 8000
EXPOSE 8501