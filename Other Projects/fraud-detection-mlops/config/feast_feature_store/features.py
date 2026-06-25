from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32, String

# 1. Definisikan Entitas
user_entity = Entity(
    name="user_id", 
    value_type=ValueType.STRING,
    description="ID dari pengguna"
)

# 2. Definisikan Sumber Data Offline (Arahkan langsung ke file Parquet)
file_source = FileSource(
    path="../../data/historical_transactions.parquet", # <--- Menggunakan Parquet
    timestamp_field="timestamp"                        # <--- Tanpa perlu argumen file_format!
)

# 3. Definisikan Feature View
user_transaction_feature_view = FeatureView(
    name="user_transaction_features",
    entities=[user_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="amount", dtype=Float32),
        Field(name="location", dtype=String),
        Field(name="device_type", dtype=String),
    ],
    online=True,
    source=file_source,
)