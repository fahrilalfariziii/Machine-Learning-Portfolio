import streamlit as st
import requests
import json

# Konfigurasi Halaman
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="centered")

st.title("🛡️ Real-Time Fraud Detection System")
st.write("Aplikasi MLOps untuk mendeteksi transaksi fraud secara instan menggunakan Feast Feature Store dan MLflow.")

st.markdown("---")

# Input Form
st.subheader("Cek Transaksi Pengguna")
user_id_input = st.text_input("Masukkan User ID:", value="user_1005")

# Tombol untuk Prediksi
if st.button("Analisis Transaksi", type="primary"):
    if user_id_input.strip() == "":
        st.warning("Silakan masukkan User ID terlebih dahulu.")
    else:
        with st.spinner("Mengambil fitur dari Feast dan mengevaluasi model..."):
            try:
                # Tembak API FastAPI (menggunakan nama service Docker atau localhost)
                # Di dalam Docker, url-nya adalah http://web-service:8000/predict
                # Untuk testing lokal di luar Docker, ganti ke http://localhost:8000/predict
                api_url = "http://web-service:8000/predict"
                # api_url = "http://localhost:8000/predict"  
                payload = {"user_id": user_id_input.strip()}
                
                response = requests.post(api_url, json=payload, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Tampilkan Status Fraud dengan Indikator Warna
                    is_fraud = result["is_fraud"]
                    prob = result["fraud_probability"]
                    
                    if is_fraud == 1:
                        st.error(f"🚨 **TERDETEKSI FRAUD!** (Probabilitas: {prob:.2%})")
                    else:
                        st.success(f"✅ **TRANSAKSI AMAN** (Probabilitas Fraud: {prob:.2%})")
                    
                    # Tampilkan Fitur yang Diambil dari Redis Store
                    st.write("### 📊 Fitur Real-Time dari Feast (Redis):")
                    features = result["features_retrieved"]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Nominal Transaksi", f"${features['amount']}:,.2f")
                    col2.metric("Lokasi", str(features['location']))
                    col3.metric("Perangkat", str(features['device_type']).title())
                    
                elif response.status_code == 404:
                    st.warning(f"⚠️ User ID `{user_id_input}` tidak ditemukan di Redis Online Store.")
                else:
                    st.error(f"❌ API Error: Status {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Gagal terhubung ke API FastAPI. Pastikan backend server `web-service` sudah berjalan.")