from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FEATURES = [
    "TX_AMOUNT",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "customer_tx_count",
    "customer_avg_amount",
    "customer_max_amount",
    "customer_amount_std",
    "time_since_customer_tx",
    "customer_amount_deviation",
    "customer_amount_ratio",
    "customer_tx_count_1h",
    "customer_tx_count_24h",
    "customer_amount_sum_24h",
    "terminal_tx_count",
    "terminal_avg_amount",
    "terminal_max_amount",
    "terminal_amount_std",
    "terminal_fraud_count",
    "terminal_fraud_rate",
    "terminal_tx_count_1h",
    "terminal_tx_count_24h",
    "terminal_amount_sum_24h",
]

validation_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_features.parquet"
)

output_path = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "sample_batch_transactions.csv"
)

validation_data = pd.read_parquet(validation_path)

missing_features = [
    feature
    for feature in MODEL_FEATURES
    if feature not in validation_data.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

sample_batch = validation_data[MODEL_FEATURES].head(5).copy()

non_negative_features = [
    "TX_AMOUNT",
    "customer_tx_count",
    "customer_avg_amount",
    "customer_max_amount",
    "customer_amount_std",
    "time_since_customer_tx",
    "customer_tx_count_1h",
    "customer_tx_count_24h",
    "customer_amount_sum_24h",
    "terminal_tx_count",
    "terminal_avg_amount",
    "terminal_max_amount",
    "terminal_amount_std",
    "terminal_fraud_count",
    "terminal_tx_count_1h",
    "terminal_tx_count_24h",
    "terminal_amount_sum_24h",
]

sample_batch[non_negative_features] = (
    sample_batch[non_negative_features]
    .clip(lower=0)
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

sample_batch.to_csv(
    output_path,
    index=False,
)

print(f"Sample batch CSV created: {output_path}")
print(f"Transactions: {len(sample_batch)}")
print(f"Features: {len(sample_batch.columns)}")