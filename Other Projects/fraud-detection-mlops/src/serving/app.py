import os
import numpy as np
import pandas as pd
import mlflow.lightgbm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore
from sklearn.preprocessing import LabelEncoder

# Inisialisasi FastAPI
app = FastAPI(title="Real-Time MLOps Fraud Detection API", version="1.0")

# Inisialisasi Feast Feature Store
# Karena app.py jalan di src/serving, kita arahkan ke folder repo Feast
STORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/feast_feature_store"))
store = FeatureStore(repo_path=STORE_PATH)


# Tunjuk URI tracking MLflow ke container mlflow-tracker (bisa via env variable)  
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Load Model Terbaik Langsung dari MLflow Registry
# Di produksi, biasanya kita menunjuk ke tag 'Production' atau versi spesifik
try:
    model_uri = "models:/fraud_lightgbm_model/1"  # Mengambil versi 1 yang baru saja Anda buat
    print(f"⏳ Memuat model dari MLflow: {model_uri}...")
    model = mlflow.lightgbm.load_model(model_uri)
    print("🚀 Model sukses dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model dari MLflow: {e}")
    model = None

# Skema Request Input dari Client
class TransactionRequest(BaseModel):
    user_id: str

# Mock Encoder (Harus sama dengan encoding saat training fase 2)
# Di sistem production nyata, encoder ini disimpan sebagai artifact di MLflow.
def mock_categorical_encoding(location, device_type):
    # Menggunakan mapping sederhana agar cepat untuk demonstrasi portofolio
    loc_encoded = hash(location) % 100
    dev_encoded = 0 if device_type == "mobile" else (1 if device_type == "desktop" else 2)
    return loc_encoded, dev_encoded

@app.get("/")
def home():
    return {"message": "Fraud Detection API is Running"}

@app.post("/predict")
def predict_fraud(payload: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model MLflow tidak siap.")
    
    user_id = payload.user_id
    
    # 1. Ambil Fitur Real-Time dari Redis via Feast
    feature_ids = [
        "user_transaction_features:amount",
        "user_transaction_features:location",
        "user_transaction_features:device_type"
    ]
    
    try:
        feast_response = store.get_online_features(
            features=feature_ids,
            entity_rows=[{"user_id": user_id}]
        ).to_dict()
        
        # Ekstrak nilai fitur (jika None/tidak ditemukan, beri default atau handling)
        amount = feast_response["amount"][0]
        location = feast_response["location"][0]
        device_type = feast_response["device_type"][0]
        
        if amount is None:
            raise HTTPException(status_code=404, detail=f"Data fitur untuk {user_id} tidak ditemukan di Redis Online Store.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengambil fitur dari Feature Store: {str(e)}")

    # 2. Preprocessing & Encoding Data Online
    loc_encoded, dev_encoded = mock_categorical_encoding(location, device_type)
    
    # Bungkus ke format matriks input yang diharapkan LightGBM [[amount, location, device_type]]
    input_data = np.array([[amount, loc_encoded, dev_encoded]])
    
    # 3. Model Inference
    prediction_prob = model.predict(input_data)[0]
    is_fraud = 1 if prediction_prob > 0.5 else 0
    
    return {
        "user_id": user_id,
        "features_retrieved": {
            "amount": amount,
            "location": location,
            "device_type": device_type
        },
        "fraud_probability": float(prediction_prob),
        "is_fraud": is_fraud
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)