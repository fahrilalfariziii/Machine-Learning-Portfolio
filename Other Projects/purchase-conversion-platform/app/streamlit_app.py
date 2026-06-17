import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

# 1. LOAD MODEL
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", "purchase_conversion_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'purchase_conversion_catboost.pkl' berada di direktori yang benar. Error: {e}")
    st.stop()

# 2. CONFIGURASI HALAMAN
st.set_page_config(
    page_title="Purchase Conversion Predictor",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Purchase Conversion Prediction Platform")
st.write("Aplikasi ini memprediksi peluang interaksi pengguna e-commerce berujung pada transaksi pembelian.")

st.markdown("---")

# 3. INTERFACE FORM INPUT USER
st.subheader("📊 Data Interaksi & Perilaku Pengguna")

col1, col2 = st.columns(2)

with col1:
    interaction_type = st.selectbox(
        "Jenis Interaksi (interaction_type)",
        options=["view", "click", "add_to_wishlist", "add_to_cart"]
    )
    dwell_time_ms = st.number_input("Durasi Sesi / Dwell Time (ms)", min_value=0, value=5000, step=500)
    device_type = st.selectbox("Tipe Perangkat (device_type)", options=["Mobile", "Desktop", "Tablet", "MISSING"])
    referrer_source = st.selectbox("Sumber Referrer", options=["Search", "Social", "Direct", "Paid", "MISSING"])

with col2:
    user_prev_interactions = st.number_input("Total Interaksi User Sebelumnya", min_value=0, value=5)
    user_prev_purchase_count = st.number_input("Total Pembelian User Sebelumnya", min_value=0, value=0)
    product_prev_interactions = st.number_input("Total Interaksi Produk Ini Sebelumnya", min_value=0, value=12)
    product_prev_purchase_count = st.number_input("Total Pembelian Produk Ini Sebelumnya", min_value=0, value=1)

st.subheader("👤 Profil Pelanggan & Produk")
col3, col4 = st.columns(2)

with col3:
    loyalty_tier = st.selectbox("Loyalitas Pelanggan", options=["Regular", "Silver", "Gold", "MISSING"])
    customer_days = st.number_input("Umur Akun Pelanggan (Hari)", min_value=0, value=120)
    category_match = st.radio("Kategori Produk Sesuai Preferensi User?", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")

with col4:
    rating_avg = st.slider("Rata-rata Rating Produk", min_value=1.0, max_value=5.0, value=4.2, step=0.1)
    review_count = st.number_input("Jumlah Review Produk", min_value=0, value=25)
    
    # Fitur Waktu Otomatis (Mengambil waktu sekarang sebagai simulasi)
    hour = st.slider("Jam Interaksi (0-23)", 0, 23, 14)
    dayofweek = st.slider("Hari (0=Senin, 6=Minggu)", 0, 6, 2)
    month = st.slider("Bulan (1-12)", 1, 12, 6)
    weekend = 1 if dayofweek >= 5 else 0

# 4. PROSES PREDIKSI
st.markdown("---")

if st.button("🚀 Hitung Probabilitas Konversi", type="primary"):
    
    # Bungkus data ke format JSON sesuai skema Pydantic di FastAPI
    payload = {
        "interaction_type": interaction_type,
        "dwell_time_ms": int(dwell_time_ms),
        "device_type": device_type,
        "referrer_source": referrer_source,
        "loyalty_tier": loyalty_tier,
        "customer_days": int(customer_days),
        "category_match": int(category_match),
        "hour": int(hour),
        "dayofweek": int(dayofweek),
        "month": int(month),
        "weekend": int(weekend),
        "user_prev_interactions": int(user_prev_interactions),
        "product_prev_interactions": int(product_prev_interactions),
        "user_prev_purchase_count": int(user_prev_purchase_count),
        "product_prev_purchase_count": int(product_prev_purchase_count),
        "rating_avg": float(rating_avg),
        "review_count": int(review_count)
    }
    
    # URL Server FastAPI (Lokal)
    FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict")
    
    try:
        # Kirim request ke FastAPI
        response = requests.post(FASTAPI_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prob = result["probability"]
            decision = result["decision"]
            
            # Tampilkan Hasil
            st.subheader("🎯 Hasil Analisis Model (via FastAPI)")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric(label="Skor Probabilitas Pembelian", value=f"{prob * 100:.2f}%")
            
            if decision == "POTENSI TINGGI":
                col_res2.success(f"🔥 {decision} (REKOMENDASI: Kirim Promosi/Diskon)")
            else:
                col_res2.warning(f"⏳ {decision} (REKOMENDASI: Simpan Anggaran Pemasaran)")
        else:
            st.error(f"API mengembalikan error: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"Gagal terhubung ke server FastAPI. Pastikan FastAPI sudah dijalankan. Error: {e}")