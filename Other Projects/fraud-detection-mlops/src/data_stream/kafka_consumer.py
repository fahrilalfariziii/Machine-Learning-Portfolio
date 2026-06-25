import json
import os
import pandas as pd
from kafka import KafkaConsumer

TOPIC_NAME = 'transactions'
DATA_DIR = 'data/'
OUTPUT_FILE = os.path.join(DATA_DIR, 'historical_transactions.csv')

# Pastikan folder data/ sudah ada
os.makedirs(DATA_DIR, exist_ok=True)

# Inisialisasi Kafka Consumer
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',  
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))  # <--- Ubah serializer menjadi deserializer
)

def save_to_csv(records):
    """Menyimpan kumpulan data ke file CSV historis"""
    df_new = pd.DataFrame(records)
    
    # Jika file sudah ada, append tanpa tulis header lagi
    if os.path.exists(OUTPUT_FILE):
        df_new.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Berhasil menyimpan {len(records)} data baru ke {OUTPUT_FILE}")

if __name__ == "__main__":
    print("📥 Kafka Consumer mulai mendengarkan data... (Tekan Ctrl+C untuk berhenti)")
    
    buffer = []
    buffer_size = 10  # Simpan ke CSV setiap 10 data terkumpul agar hemat I/O disk
    
    try:
        for message in consumer:
            tx_data = message.value
            buffer.append(tx_data)
            print(f"📥 Diterima: {tx_data['transaction_id']} | User: {tx_data['user_id']}")
            
            if len(buffer) >= buffer_size:
                save_to_csv(buffer)
                buffer = []  # Kosongkan buffer
                
    except KeyboardInterrupt:
        print("\n🛑 Consumer dihentikan.")
        if buffer:
            save_to_csv(buffer)  # Simpan sisa data yang ada di buffer