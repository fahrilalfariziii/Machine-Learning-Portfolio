from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Purchase Conversion Prediction API",
    description="API untuk memprediksi probabilitas konversi pembelian pengguna e-commerce.",
    version="1.0"
)

# 1. LOAD MODEL (Aman terhadap path)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "models", "purchase_conversion_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    model_features = model.feature_names_
    cat_features = model.get_cat_feature_indices()
    cat_feature_names = [model_features[i] for i in cat_features]
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Gagal memuat model CatBoost: {e}")
    #raise RuntimeDataError(f"Gagal memuat model CatBoost: {e}")

# 2. DEFINISIKAN SCHEMA INPUT (Menggunakan Pydantic untuk validasi data)
class InteractionInput(BaseModel):
    interaction_type: str
    dwell_time_ms: int
    device_type: str
    referrer_source: str
    loyalty_tier: str
    customer_days: int
    category_match: int
    hour: int
    dayofweek: int
    month: int
    weekend: int
    user_prev_interactions: int
    product_prev_interactions: int
    user_prev_purchase_count: int
    product_prev_purchase_count: int
    rating_avg: float
    review_count: int

# 3. ENDPOINT UTAMA
@app.get("/")
def home():
    return {"message": "Purchase Conversion Prediction API is running!"}

# 4. ENDPOINT PREDIKSI
@app.post("/predict")
def predict_conversion(input_item: InteractionInput):
    try:
        # Konversi input data JSON menjadi dictionary
        raw_data = input_item.dict()
        
        # Hitung fitur turunan yang dibutuhkan model secara otomatis
        raw_data["user_avg_dwell_before"] = float(raw_data["dwell_time_ms"])
        raw_data["product_avg_dwell_before"] = float(raw_data["dwell_time_ms"])
        raw_data["user_hist_conversion"] = raw_data["user_prev_purchase_count"] / (raw_data["user_prev_interactions"] + 1)
        raw_data["product_hist_conversion"] = raw_data["product_prev_purchase_count"] / (raw_data["product_prev_interactions"] + 1)
        
        # Buat DataFrame sesuai urutan fitur model
        input_data = pd.DataFrame(columns=model_features)
        
        for col in model_features:
            if col in raw_data:
                input_data.loc[0, col] = raw_data[col]
            else:
                if col in cat_feature_names:
                    input_data.loc[0, col] = "MISSING"
                else:
                    input_data.loc[0, col] = 0.0

        # Cast tipe data numerik
        for col in model_features:
            if col not in cat_feature_names:
                input_data[col] = pd.to_numeric(input_data[col])

        # Jalankan Prediksi
        prob = float(model.predict_proba(input_data)[0][1])
        THRESHOLD = 0.1283
        is_purchase = prob > THRESHOLD

        return {
            "status": "success",
            "probability": prob,
            "prediction": 1 if is_purchase else 0,
            "decision": "POTENSI TINGGI" if is_purchase else "POTENSI RENDAH"
        }

    except Exception as e:
        raise HTTPException(status_index=500, detail=f"Terjadi kesalahan pada server: {str(e)}")