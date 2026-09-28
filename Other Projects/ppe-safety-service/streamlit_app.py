from pathlib import Path

import torch

_original_torch_load = torch.load


def _forced_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _forced_torch_load

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "weights" / "safety_best_model.pt"

st.set_page_config(
    page_title="SafetyVision: Enterprise PPE Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model YOLO: {e}")
    st.stop()

st.sidebar.title("SafetyVision Engine")
st.sidebar.markdown("---")
st.sidebar.info(
    "Sistem monitoring kepatuhan APD (Helm & Rompi) berbasis AI untuk "
    "skala industri, manufaktur, dan konstruksi."
)

confidence_threshold = st.sidebar.slider(
    "Ambang Batas Deteksi (Confidence)",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.caption("Tech Stack: YOLOv10, Streamlit (Cloud-native, tanpa FastAPI)")
st.sidebar.success(f"Model: {MODEL_PATH.name}")

st.title("Real-Time Industrial PPE Compliance Dashboard")
st.subheader("Modul Verifikasi Keselamatan Kerja Mandiri")
st.write("Unggah foto area kerja untuk mendeteksi penggunaan APD secara otomatis.")

uploaded_file = st.file_uploader("Pilih gambar (.jpg, .jpeg, .png)...", type=["jpg", "jpeg", "png"])

COLOR_MAP = {
    "helmet": (0, 255, 0),
    "vest": (0, 165, 255),
    "no_helmet": (255, 0, 0),
    "no_vest": (255, 0, 0),
    "person": (255, 255, 0),
}

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.markdown("### Gambar Asli (Input)")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### Analisis AI (Output)")
        with st.spinner("Memproses gambar via AI Engine..."):
            try:
                img_np = np.array(image)
                results = model.predict(source=img_np, conf=confidence_threshold, verbose=False)
                result = results[0]

                annotated = img_np.copy()
                if annotated.shape[-1] == 4:
                    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGBA2RGB)

                detections = []
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        coords = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        detections.append(
                            {
                                "object": class_name,
                                "confidence": round(confidence, 4),
                                "bounding_box": {
                                    "xmin": round(coords[0], 1),
                                    "ymin": round(coords[1], 1),
                                    "xmax": round(coords[2], 1),
                                    "ymax": round(coords[3], 1),
                                },
                            }
                        )

                for det in detections:
                    obj_name = det["object"]
                    conf = det["confidence"]
                    bbox = det["bounding_box"]
                    color = COLOR_MAP.get(obj_name, (255, 255, 255))
                    cv2.rectangle(
                        annotated,
                        (int(bbox["xmin"]), int(bbox["ymin"])),
                        (int(bbox["xmax"]), int(bbox["ymax"])),
                        color,
                        3,
                    )
                    cv2.putText(
                        annotated,
                        f"{obj_name} ({conf * 100:.1f}%)",
                        (int(bbox["xmin"]), int(bbox["ymin"]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                st.image(annotated, use_container_width=True)
                st.success(f"Analisis Selesai! Menemukan {len(detections)} deteksi valid.")

                with st.expander("Lihat Respon Data Mentah (JSON)"):
                    st.json(
                        {
                            "success": True,
                            "total_detections": len(detections),
                            "detections": detections,
                        }
                    )
            except Exception as e:
                st.error(f"Gagal menjalankan inferensi: {e}")
