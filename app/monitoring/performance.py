from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core import (
    PerformanceEvaluationError,
    get_logger,
)


logger = get_logger(__name__)


def calculate_performance_metrics(
    actual_labels: list[int],
    predicted_labels: list[int],
    fraud_probabilities: list[float] | None = None,
) -> dict[str, Any]:
    """
    Calculate production classification performance metrics.

    Parameters
    ----------
    actual_labels:
        Observed transaction outcomes.

    predicted_labels:
        Model fraud predictions.

    fraud_probabilities:
        Optional predicted fraud probabilities used to
        calculate ROC-AUC.

    Returns
    -------
    dict[str, Any]
        Production classification performance metrics.

    Raises
    ------
    PerformanceEvaluationError
        If the provided evaluation data is invalid.
    """

    if not actual_labels:
        raise PerformanceEvaluationError(
            "actual_labels cannot be empty."
        )

    if len(actual_labels) != len(
        predicted_labels
    ):
        raise PerformanceEvaluationError(
            "actual_labels and predicted_labels "
            "must have the same length."
        )

    if fraud_probabilities is not None:
        if len(actual_labels) != len(
            fraud_probabilities
        ):
            raise PerformanceEvaluationError(
                "actual_labels and fraud_probabilities "
                "must have the same length."
            )

    actual_array = np.asarray(
        actual_labels
    )

    predicted_array = np.asarray(
        predicted_labels
    )

    metrics: dict[str, Any] = {
        "total_transactions": int(
            len(actual_array)
        ),
        "actual_fraud_count": int(
            np.sum(actual_array == 1)
        ),
        "predicted_fraud_count": int(
            np.sum(predicted_array == 1)
        ),
        "accuracy": round(
            float(
                accuracy_score(
                    actual_array,
                    predicted_array,
                )
            ),
            6,
        ),
        "precision": round(
            float(
                precision_score(
                    actual_array,
                    predicted_array,
                    zero_division=0,
                )
            ),
            6,
        ),
        "recall": round(
            float(
                recall_score(
                    actual_array,
                    predicted_array,
                    zero_division=0,
                )
            ),
            6,
        ),
        "f1_score": round(
            float(
                f1_score(
                    actual_array,
                    predicted_array,
                    zero_division=0,
                )
            ),
            6,
        ),
    }

    if fraud_probabilities is not None:
        probability_array = np.asarray(
            fraud_probabilities
        )

        if len(
            np.unique(actual_array)
        ) > 1:
            metrics["roc_auc"] = round(
                float(
                    roc_auc_score(
                        actual_array,
                        probability_array,
                    )
                ),
                6,
            )
        else:
            metrics["roc_auc"] = None

    logger.info(
        "Performance metrics calculated | "
        "total_transactions=%s | "
        "accuracy=%.6f | "
        "precision=%.6f | "
        "recall=%.6f | "
        "f1_score=%.6f | "
        "roc_auc=%s",
        metrics["total_transactions"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics.get("roc_auc"),
    )

    return metrics