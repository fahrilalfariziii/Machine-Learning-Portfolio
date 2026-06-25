import os
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load Data Historis
PARQUET_PATH = "data/historical_transactions.parquet"
if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(f"Data {PARQUET_PATH} tidak ditemukan. Selesaikan Fase 1 dulu!")

df = pd.read_parquet(PARQUET_PATH)

# Preprocessing Sederhana (Encoding untuk Fitur Kategorikal)
le_location = LabelEncoder()
df['location'] = le_location.fit_transform(df['location'])

le_device = LabelEncoder()
df['device_type'] = le_device.fit_transform(df['device_type'])

# Tentukan Fitur dan Target
X = df[['amount', 'location', 'device_type']]
y = df['is_fraud']

# Split Data
# Perbaikan dari test_test_split menjadi test_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Set up MLflow Experiment Name
mlflow.set_experiment("Fraud_Detection_Optimization")

def objective(trial):
    """Fungsi objektif untuk Optuna Hyperparameter Tuning"""
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 20),
        "verbosity": -1
    }
    
    # Train dengan parameter trial
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    return auc

if __name__ == "__main__":
    print("🧪 Memulai Hyperparameter Tuning dengan Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10) # 10 kali percobaan percobaan parameter
    
    print(f"🏆 Parameter Terbaik: {study.best_params}")
    
    # 2. Train Model Final dengan Parameter Terbaik & Track ke MLflow
    with mlflow.start_run(run_name="Best_LightGBM_Model"):
        print("📊 Melatih model final dan mencatat ke MLflow...")
        
        # Log parameter terbaik dari Optuna ke MLflow
        mlflow.log_params(study.best_params)
        
        best_params = study.best_params
        best_params["objective"] = "binary"
        best_params["metric"] = "auc"
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        best_model = lgb.train(
            best_params,
            train_data,
            valid_sets=[val_data]
        )
        
        # Evaluasi Model
        preds = best_model.predict(X_test)
        preds_binary = [1 if p > 0.5 else 0 for p in preds]
        
        auc_score = roc_auc_score(y_test, preds)
        f1 = f1_score(y_test, preds_binary, zero_division=0)
        
        # Log Metrik ke MLflow
        mlflow.log_metric("auc", auc_score)
        mlflow.log_metric("f1_score", f1)
        
        # Log Model sebagai Artifact
        mlflow.lightgbm.log_model(
            lgb_model=best_model,
            artifact_path="model",
            registered_model_name="fraud_lightgbm_model"
        )
        
        print(f"✅ Model berhasil di-track! AUC: {auc_score:.4f} | F1: {f1:.4f}")