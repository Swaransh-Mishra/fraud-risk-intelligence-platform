import pandas as pd


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer behavioural features using only transaction history
    available before the current transaction.
    """

    data = df.copy()

    data["_original_order"] = range(len(data))

    # Customer behaviour must be evaluated in chronological order.
    data = data.sort_values(
        ["CUSTOMER_ID", "TX_DATETIME", "TRANSACTION_ID"]
    ).reset_index(drop=True)

    customer_group = data.groupby("CUSTOMER_ID", sort=False)

    # Historical transaction count before the current transaction.
    data["customer_tx_count"] = customer_group.cumcount()

    # Historical spending behaviour based only on previous transactions.
    data["customer_avg_amount"] = (
        customer_group["TX_AMOUNT"]
        .transform(lambda series: series.shift(1).expanding().mean())
    )

    data["customer_max_amount"] = (
        customer_group["TX_AMOUNT"]
        .transform(lambda series: series.shift(1).expanding().max())
    )

    data["customer_amount_std"] = (
        customer_group["TX_AMOUNT"]
        .transform(lambda series: series.shift(1).expanding().std())
    )

    # Time since the customer's previous transaction.
    previous_transaction_time = customer_group["TX_DATETIME"].shift(1)

    data["time_since_customer_tx"] = (
        data["TX_DATETIME"] - previous_transaction_time
    ).dt.total_seconds()

    # Current transaction amount relative to the customer's historical behaviour.
    data["customer_amount_deviation"] = (
        data["TX_AMOUNT"] - data["customer_avg_amount"]
    )

    data["customer_amount_ratio"] = (
        data["TX_AMOUNT"]
        / data["customer_avg_amount"].replace(0, pd.NA)
    )

    # Recent customer activity is calculated transaction-by-transaction
    # using a deque to ensure only previous transactions are included.
    customer_tx_count_1h = []
    customer_tx_count_24h = []
    customer_amount_sum_24h = []

    for _, customer_data in data.groupby("CUSTOMER_ID", sort=False):
        history = []

        for _, row in customer_data.iterrows():
            current_time = row["TX_DATETIME"]
            current_amount = row["TX_AMOUNT"]

            one_hour_start = current_time - pd.Timedelta(hours=1)
            twenty_four_hour_start = current_time - pd.Timedelta(hours=24)

            recent_1h = [
                transaction
                for transaction in history
                if transaction[0] > one_hour_start
            ]

            recent_24h = [
                transaction
                for transaction in history
                if transaction[0] > twenty_four_hour_start
            ]

            customer_tx_count_1h.append(len(recent_1h))

            customer_tx_count_24h.append(len(recent_24h))

            customer_amount_sum_24h.append(
                sum(transaction[1] for transaction in recent_24h)
            )

            history.append((current_time, current_amount))

    data["customer_tx_count_1h"] = customer_tx_count_1h
    data["customer_tx_count_24h"] = customer_tx_count_24h
    data["customer_amount_sum_24h"] = customer_amount_sum_24h

    data = (
         data
         .sort_values("_original_order")
         .drop(columns="_original_order")
         .reset_index(drop=True)
   )

    return data