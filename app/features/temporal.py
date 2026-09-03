from __future__ import annotations

import pandas as pd


def add_temporal_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add transaction-time features used by the fraud model."""
    required_columns = {"TX_DATETIME"}

    missing_columns = required_columns.difference(transactions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    features = transactions.copy()

    if not pd.api.types.is_datetime64_any_dtype(features["TX_DATETIME"]):
        features["TX_DATETIME"] = pd.to_datetime(
            features["TX_DATETIME"],
            errors="raise",
        )

    features["hour_of_day"] = features["TX_DATETIME"].dt.hour
    features["day_of_week"] = features["TX_DATETIME"].dt.dayofweek
    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype("int8")

    return features