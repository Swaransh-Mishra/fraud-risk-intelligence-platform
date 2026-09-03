from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


LOG_DIRECTORY = Path("logs")

LOG_FILE_PATH = (
    LOG_DIRECTORY
    / "prediction_events.jsonl"
)


def log_prediction_event(
    event_type: str,
    model_name: str,
    decision_threshold: float,
    total_transactions: int,
    predicted_fraud_count: int,
    average_fraud_probability: float,
) -> None:
    """
    Store a prediction monitoring event in JSONL format.

    Each line in the log file represents one prediction
    or batch prediction event.
    """

    LOG_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    event = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "event_type": event_type,
        "model_name": model_name,
        "decision_threshold": float(
            decision_threshold
        ),
        "total_transactions": int(
            total_transactions
        ),
        "predicted_fraud_count": int(
            predicted_fraud_count
        ),
        "average_fraud_probability": float(
            average_fraud_probability
        ),
    }

    with LOG_FILE_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(event)
            + "\n"
        )