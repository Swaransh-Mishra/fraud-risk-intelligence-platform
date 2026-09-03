from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


LOG_FILE_PATH = (
    Path("logs")
    / "prediction_events.jsonl"
)


def load_monitoring_events() -> pd.DataFrame:
    """
    Load prediction monitoring events from the JSONL log file.
    """

    if not LOG_FILE_PATH.exists():
        return pd.DataFrame()

    events = []

    with LOG_FILE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            events.append(
                json.loads(line)
            )

    if not events:
        return pd.DataFrame()

    events_df = pd.DataFrame(events)

    events_df["timestamp"] = pd.to_datetime(
        events_df["timestamp"],
        utc=True,
    )

    return events_df


def get_monitoring_report() -> dict:
    """
    Generate aggregated monitoring analytics
    from production prediction events.
    """

    events_df = load_monitoring_events()

    if events_df.empty:
        return {
            "summary": {
                "total_events": 0,
                "total_transactions": 0,
                "total_predicted_fraud": 0,
                "fraud_alert_rate": 0.0,
                "average_fraud_probability": 0.0,
            },
            "event_type_summary": [],
            "model_activity_summary": [],
            "latest_events": [],
        }

    total_events = len(events_df)

    total_transactions = int(
        events_df["total_transactions"].sum()
    )

    total_predicted_fraud = int(
        events_df[
            "predicted_fraud_count"
        ].sum()
    )

    fraud_alert_rate = (
        total_predicted_fraud
        / total_transactions
        if total_transactions > 0
        else 0.0
    )

    average_fraud_probability = float(
        events_df[
            "average_fraud_probability"
        ].mean()
    )

    event_type_summary = (
        events_df
        .groupby(
            "event_type",
            as_index=False,
        )
        .agg(
            event_count=(
                "event_type",
                "size",
            ),
            total_transactions=(
                "total_transactions",
                "sum",
            ),
            predicted_fraud_count=(
                "predicted_fraud_count",
                "sum",
            ),
            average_fraud_probability=(
                "average_fraud_probability",
                "mean",
            ),
        )
        .to_dict(
            orient="records"
        )
    )

    model_activity_summary = (
        events_df
        .groupby(
            "model_name",
            as_index=False,
        )
        .agg(
            event_count=(
                "model_name",
                "size",
            ),
            total_transactions=(
                "total_transactions",
                "sum",
            ),
            predicted_fraud_count=(
                "predicted_fraud_count",
                "sum",
            ),
            average_fraud_probability=(
                "average_fraud_probability",
                "mean",
            ),
        )
        .to_dict(
            orient="records"
        )
    )

    latest_events_df = (
        events_df
        .iloc[::-1]
        .head(10)
    )

    latest_events = []

    for _, row in latest_events_df.iterrows():
        latest_events.append(
            {
                "timestamp": (
                    row["timestamp"]
                    .isoformat()
                ),
                "event_type": str(
                    row["event_type"]
                ),
                "model_name": str(
                    row["model_name"]
                ),
                "decision_threshold": float(
                    row[
                        "decision_threshold"
                    ]
                ),
                "total_transactions": int(
                    row[
                        "total_transactions"
                    ]
                ),
                "predicted_fraud_count": int(
                    row[
                        "predicted_fraud_count"
                    ]
                ),
                "average_fraud_probability": float(
                    row[
                        "average_fraud_probability"
                    ]
                ),
            }
        )

    return {
        "summary": {
            "total_events": int(
                total_events
            ),
            "total_transactions": int(
                total_transactions
            ),
            "total_predicted_fraud": int(
                total_predicted_fraud
            ),
            "fraud_alert_rate": round(
                float(fraud_alert_rate),
                6,
            ),
            "average_fraud_probability": round(
                average_fraud_probability,
                6,
            ),
        },
        "event_type_summary": (
            event_type_summary
        ),
        "model_activity_summary": (
            model_activity_summary
        ),
        "latest_events": latest_events,
    }