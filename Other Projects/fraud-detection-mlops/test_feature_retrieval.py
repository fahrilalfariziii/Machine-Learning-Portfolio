from feast import FeatureStore
import pandas as pd
import os

parquet_path = "data/historical_transactions.parquet"

if not os.path.exists(parquet_path):
    print(f"❌ File {parquet_path} tidak ditemukan! Selesaikan konversi ke Parquet terlebih dahulu.")
    exit()

# 1. Ambil sampel user_id yang PASTI ADA di dalam data Anda
df_parquet = pd.read_parquet(parquet_path)
if df_parquet.empty:
    print("❌ File Parquet Anda kosong. Jalankan Kafka Producer & Consumer lebih lama terlebih dahulu.")
    exit()

sample_user_id = str(df_parquet['user_id'].iloc[0]) # Mengambil user pertama dari data asli
print(f"🎯 Menemukan user valid di dalam database: {sample_user_id}")

# 2. Inisialisasi Feature Store
store = FeatureStore(repo_path="config/feast_feature_store")

feature_ids = [
    "user_transaction_features:amount",
    "user_transaction_features:location",
    "user_transaction_features:device_type"
]

entity_rows = [{"user_id": sample_user_id}]

print(f"🔍 Mengambil fitur untuk {sample_user_id} dari Redis (Online Store)...")

try:
    response = store.get_online_features(
        features=feature_ids,
        entity_rows=entity_rows
    ).to_dict()

    df_features = pd.DataFrame(response)
    print("\n✅ Fitur Berhasil Diambil:")
    print(df_features.to_string(index=False))

except Exception as e:
    print(f"❌ Gagal mengambil data: {e}")