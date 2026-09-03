from collections import deque

import numpy as np
import pandas as pd


def add_terminal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create terminal-level historical, recent-activity, and fraud-history features.

    All historical features use information available before the current
    transaction to avoid target leakage.
    """

    required_columns = {
        "TRANSACTION_ID",
        "TERMINAL_ID",
        "TX_DATETIME",
        "TX_AMOUNT",
        "TX_FRAUD",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    features = df.copy()

    features["TX_DATETIME"] = pd.to_datetime(
        features["TX_DATETIME"]
    )

    # Preserve the original row order before sorting by terminal and time.
    features["_original_order"] = np.arange(len(features))

    features = features.sort_values(
        ["TERMINAL_ID", "TX_DATETIME", "TRANSACTION_ID"]
    ).reset_index(drop=True)

    terminal_group = features.groupby(
        "TERMINAL_ID",
        sort=False
    )

    # ------------------------------------------------------------------
    # Historical terminal behavior
    # ------------------------------------------------------------------

    features["terminal_tx_count"] = terminal_group.cumcount()

    previous_amount = terminal_group["TX_AMOUNT"].shift(1)

    features["terminal_avg_amount"] = (
        previous_amount
        .groupby(features["TERMINAL_ID"], sort=False)
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    features["terminal_max_amount"] = (
        previous_amount
        .groupby(features["TERMINAL_ID"], sort=False)
        .cummax()
    )

    features["terminal_amount_std"] = (
        previous_amount
        .groupby(features["TERMINAL_ID"], sort=False)
        .expanding()
        .std()
        .reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------------
    # Historical fraud behavior
    # ------------------------------------------------------------------

    previous_fraud = terminal_group["TX_FRAUD"].shift(1)

    features["terminal_fraud_count"] = (
        previous_fraud
        .fillna(0)
        .groupby(features["TERMINAL_ID"], sort=False)
        .cumsum()
    )

    previous_transaction_count = features["terminal_tx_count"].replace(
        0,
        np.nan,
    )

    features["terminal_fraud_rate"] = (
        features["terminal_fraud_count"]
        / previous_transaction_count
    )

    # ------------------------------------------------------------------
    # Recent terminal activity
    # ------------------------------------------------------------------

    features["terminal_tx_count_1h"] = 0
    features["terminal_tx_count_24h"] = 0
    features["terminal_amount_sum_24h"] = 0.0

    one_hour = 60 * 60
    twenty_four_hours = 24 * 60 * 60

    for _, group_index in features.groupby(
        "TERMINAL_ID",
        sort=False
    ).groups.items():

        indices = list(group_index)

        transactions_1h = deque()
        transactions_24h = deque()

        amount_window_24h = deque()
        amount_sum_24h = 0.0

        for index in indices:

            current_time = features.at[
                index,
                "TX_DATETIME"
            ].timestamp()

            # Remove transactions outside the one-hour window.
            while (
                transactions_1h
                and current_time - transactions_1h[0] > one_hour
            ):
                transactions_1h.popleft()

            # Remove transactions outside the 24-hour count window.
            while (
                transactions_24h
                and current_time - transactions_24h[0] > twenty_four_hours
            ):
                transactions_24h.popleft()

            # Remove transactions outside the 24-hour amount window.
            while (
                amount_window_24h
                and current_time - amount_window_24h[0][0]
                > twenty_four_hours
            ):
                _, expired_amount = amount_window_24h.popleft()
                amount_sum_24h -= expired_amount

            # Features are calculated before the current transaction
            # is added to each history window.
            features.at[
                index,
                "terminal_tx_count_1h"
            ] = len(transactions_1h)

            features.at[
                index,
                "terminal_tx_count_24h"
            ] = len(transactions_24h)

            features.at[
                index,
                "terminal_amount_sum_24h"
            ] = amount_sum_24h

            current_amount = features.at[
                index,
                "TX_AMOUNT"
            ]

            transactions_1h.append(current_time)
            transactions_24h.append(current_time)

            amount_window_24h.append(
                (current_time, current_amount)
            )

            amount_sum_24h += current_amount

    # Restore original transaction order.
    features = (
        features
        .sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )

    return features