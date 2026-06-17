import io
import os

# =====================================================================
# 🚨 PYTORCH 2.6+ RADICAL MONKEY PATCH (WINDOWS MULTIPROCESSING FIX)
# Membajak fungsi torch.load sebelum diimpor oleh Ultralytics agar
# parameter weights_only dipaksa bernilai False di semua sub-proses.
# =====================================================================
import torch

original_torch_load = torch.load

def forced_torch_load(*args, **kwargs):
    # Paksa parameter weights_only menjadi False
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)

# Ganti fungsi asli dengan fungsi bajakan kita
torch.load = forced_torch_load
# =====================================================================

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# 1. Inisialisasi FastAPI
app = FastAPI(
    title="Industrial PPE Detection Service",
    description="API End-to-End untuk mendeteksi helm dan rompi keselamatan kerja menggunakan YOLOv10.",
    version="1.0.0"
)

# 2. Muat Model yang Anda Download
MODEL_PATH = "weights/safety_best_model.pt" 
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model dari {MODEL_PATH}. Error: {str(e)}")

@app.get("/")
async def root():
    """Endpoint cek kesehatan server (Health Check)."""
    return {"status": "online", "message": "PPE Detection Service siap menerima request."}

@app.post("/predict")
async def predict_ppe(file: UploadFile = File(...)):
    """
    Endpoint Utama untuk Deteksi APD.
    Menerima file gambar dan mengembalikan koordinat bounding box beserta kelasnya.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Format file harus berupa JPEG atau PNG.")
    
    try:
        request_object_content = await file.read()
        image_bytes = io.BytesIO(request_object_content)
        
        file_bytes = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="File gambar rusak atau tidak dapat dibaca.")

        # Jalankan Inferensi
        results = model.predict(source=img, conf=0.25, verbose=False)
        result = results[0]
        
        detections = []
        boxes = result.boxes
        
        for box in boxes:
            coords = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            detections.append({
                "object": class_name,
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "xmin": round(coords[0], 1),
                    "ymin": round(coords[1], 1),
                    "xmax": round(coords[2], 1),
                    "ymax": round(coords[3], 1)
                }
            })
            
        return JSONResponse(content={
            "success": True,
            "total_detections": len(detections),
            "detections": detections
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Terjadi kesalahan internal: {str(e)}"}
        )