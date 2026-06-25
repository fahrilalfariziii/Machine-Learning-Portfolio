import pandas as pd
import os

csv_path = "data/historical_transactions.csv"
parquet_path = "data/historical_transactions.parquet"

if os.path.exists(csv_path):
    print("⏳ Membaca file CSV...")
    df = pd.read_csv(csv_path)
    
    # Feast WAJIB membutuhkan kolom timestamp bertipe datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("📦 Mengonversi dan menyimpan ke format Parquet...")
    df.to_parquet(parquet_path, index=False)
    print(f"✅ Sukses! File Parquet berhasil dibuat di: {parquet_path}")
else:
    print(f"❌ File {csv_path} tidak ditemukan. Pastikan Anda sudah menjalankan consumer sebelumnya.")