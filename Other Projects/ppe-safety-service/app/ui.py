import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman Dashboard
st.set_page_config(
    page_title="SafetyVision: Enterprise PPE Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konfigurasi URL Backend FastAPI
# Jika dijalankan lokal, biasanya menggunakan localhost:8000
FASTAPI_URL = "http://127.0.0.1:8000/predict"

# 2. Desain Sidebar (Kontrol & Informasi)
st.sidebar.title("🛡️ SafetyVision Engine")
st.sidebar.markdown("---")
st.sidebar.info(
    "Sistem monitoring kepatuhan APD (Helm & Rompi) berbasis AI untuk "
    "skala industri, manufaktur, dan konstruksi."
)

confidence_threshold = st.sidebar.slider(
    "Ambang Batas Deteksi (Confidence)", 
    min_value=0.1, max_value=1.0, value=0.25, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.caption("Tech Stack: YOLOv10, FastAPI, Streamlit, Docker")

# 3. Konten Utama Dashboard
st.title("🏭 Real-Time Industrial PPE Compliance Dashboard")
st.subheader("Modul Verifikasi Keselamatan Kerja Mandiri")
st.write("Unggah foto area kerja untuk mendeteksi penggunaan APD secara otomatis.")

# Komponen Pengunggah File Gambar
uploaded_file = st.file_uploader("Pilih gambar (.jpg, .jpeg, .png)...", type=["jpg", "jpeg", "png"])

# 4. Logika Pemrosesan Gambar
if uploaded_file is not None:
    # Tampilkan layout 2 kolom: Gambar Asli vs Gambar Hasil Deteksi
    col1, col2 = st.columns(2)
    
    # Baca gambar yang diunggah menggunakan PIL
    image = Image.open(uploaded_file)
    
    with col1:
        st.markdown("### 📸 Gambar Asli (Input)")
        st.image(image, use_column_width=True)
        
    with col2:
        st.markdown("### 🤖 Analisis AI (Output)")
        
        # Siapkan file untuk dikirim via HTTP Request ke FastAPI
        # Konversi file yang diunggah ke format bytes
        img_bytes = uploaded_file.getvalue()
        files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
        
        with st.spinner("Memproses gambar via AI Engine..."):
            try:
                # Kirim permintaan POST ke FastAPI Backend
                response = requests.post(FASTAPI_URL, files=files)
                
                if response.status_code == 200:
                    result_json = response.json()
                    
                    # Konversi gambar PIL ke NumPy Array (OpenCV) untuk digambari bounding box
                    img_np = np.array(image)
                    # Jika gambar berformat RGBA, ubah ke RGB
                    if img_np.shape[-1] == 4:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                        
                    detections = result_json.get("detections", [])
                    total_detected = 0
                    
                    # Definisikan warna kotak berdasarkan kelas (Format RGB)
                    color_map = {
                        "helmet": (0, 255, 0),       # Hijau
                        "vest": (0, 165, 255),       # Oranye
                        "no_helmet": (255, 0, 0),    # Merah
                        "no_vest": (255, 0, 0),       # Merah
                        "person": (255, 255, 0)      # Kuning
                    }
                    
                    # Gambar bounding box secara dinamis berdasarkan respons JSON FastAPI
                    for det in detections:
                        conf = det["confidence"]
                        # Filter berdasarkan slider confidence di sidebar
                        if conf >= confidence_threshold:
                            total_detected += 1
                            obj_name = det["object"]
                            bbox = det["bounding_box"]
                            
                            # Gambar kotak koordinat
                            cv2.rectangle(
                                img_np, 
                                (int(bbox["xmin"]), int(bbox["ymin"])), 
                                (int(bbox["xmax"]), int(bbox["ymax"])), 
                                color_map.get(obj_name, (255, 255, 255)), 
                                3
                            )
                            
                            # Tulis label teks di atas kotak
                            label = f"{obj_name} ({conf*100:.1f}%)"
                            cv2.putText(
                                img_np, label, 
                                (int(bbox["xmin"]), int(bbox["ymin"]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                color_map.get(obj_name, (255, 255, 255)), 2
                            )
                    
                    # Tampilkan gambar hasil anotasi ke kolom 2
                    st.image(img_np, use_column_width=True)
                    
                    # Tampilkan Ringkasan Analisis Metrik Bisnis
                    st.success(f"Analisis Selesai! Menemukan {total_detected} deteksi valid.")
                    
                    # Menampilkan data JSON mentah jika user ingin melihat backend-nya
                    with st.expander("Lihat Respon Data Mentah (JSON API)"):
                        st.json(result_json)
                        
                else:
                    st.error(f"Gagal mendapatkan respons dari server AI. Status: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Gagal terhubung ke FastAPI backend! Pastikan server FastAPI Anda sudah dijalankan dengan uvicorn.")