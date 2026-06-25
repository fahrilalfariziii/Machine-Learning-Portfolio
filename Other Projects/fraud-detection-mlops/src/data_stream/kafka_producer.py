import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
from faker import Faker

# Inisialisasi Faker dan Kafka Producer
fake = Faker()
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TOPIC_NAME = 'transactions'

def generate_transaction():
    """Membuat data transaksi sintetis secara acak"""
    user_id = f"user_{random.randint(1000, 9999)}"
    amount = round(random.uniform(10.0, 5000.0), 2)
    
    # Membuat skenario fraud tiruan (misal: jika transaksi > 4000, kemungkinan fraud tinggi)
    is_fraud = 0
    if amount > 4000.0 and random.random() < 0.7:
        is_fraud = 1
        
    return {
        "transaction_id": fake.uuid4(),
        "user_id": user_id,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "amount": amount,
        "location": fake.city(),
        "device_type": random.choice(['mobile', 'desktop', 'tablet']),
        "is_fraud": is_fraud
    }

if __name__ == "__main__":
    print("🚀 Kafka Producer mulai mengirim data transaksi... (Tekan Ctrl+C untuk berhenti)")
    try:
        while True:
            tx_data = generate_transaction()
            producer.send(TOPIC_NAME, value=tx_data)
            print(f"🔹 Mengirim: {tx_data['transaction_id']} | User: {tx_data['user_id']} | Amount: ${tx_data['amount']} | Fraud: {tx_data['is_fraud']}")
            
            # Kirim data setiap 1 detik
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Producer dihentikan.")
    finally:
        producer.close()