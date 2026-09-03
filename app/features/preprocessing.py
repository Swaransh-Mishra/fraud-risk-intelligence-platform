from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_FEATURES = [
    # Transaction features
    "TX_AMOUNT",

    # Temporal features
    "hour_of_day",
    "day_of_week",
    "is_weekend",

    # Customer historical behaviour
    "customer_tx_count",
    "customer_avg_amount",
    "customer_max_amount",
    "customer_amount_std",
    "time_since_customer_tx",
    "customer_amount_deviation",
    "customer_amount_ratio",

    # Customer recent activity
    "customer_tx_count_1h",
    "customer_tx_count_24h",
    "customer_amount_sum_24h",

    # Terminal historical behaviour
    "terminal_tx_count",
    "terminal_avg_amount",
    "terminal_max_amount",
    "terminal_amount_std",

    # Terminal historical fraud behaviour
    "terminal_fraud_count",
    "terminal_fraud_rate",

    # Terminal recent activity
    "terminal_tx_count_1h",
    "terminal_tx_count_24h",
    "terminal_amount_sum_24h",
]


TARGET_COLUMN = "TX_FRAUD"


def prepare_model_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the engineered dataset for model training.

    Missing historical statistics represent cold-start cases
    where no previous behavioural history is available.
    """

    missing_columns = (
        set(MODEL_FEATURES + [TARGET_COLUMN])
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    cold_start_zero_features = [
        "customer_avg_amount",
        "customer_max_amount",
        "customer_amount_std",
        "time_since_customer_tx",
        "customer_amount_deviation",
        "customer_amount_ratio",
        "terminal_avg_amount",
        "terminal_max_amount",
        "terminal_amount_std",
        "terminal_fraud_count",
        "terminal_fraud_rate",
    ]

    X[cold_start_zero_features] = (
        X[cold_start_zero_features]
        .fillna(0)
    )

    return X, y