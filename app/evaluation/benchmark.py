from __future__ import annotations

import pandas as pd

from app.evaluation.business import (
    analyze_probability_business_thresholds,
)
from app.evaluation.metrics import evaluate_probabilities


def evaluate_benchmark(
    probabilities,
    y: pd.Series,
    threshold: float = 0.8,
    false_positive_cost: float = 5.0,
    false_negative_cost: float = 100.0,
) -> dict:
    """Evaluate probability scores using model and business metrics."""

    metrics = evaluate_probabilities(
        probabilities=probabilities,
        y=y,
        threshold=threshold,
    )

    business_results = analyze_probability_business_thresholds(
        probabilities=probabilities,
        y=y,
        thresholds=[threshold],
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    ).iloc[0]

    return {
        **metrics,
        "false_positives": int(
            business_results["false_positives"]
        ),
        "false_negatives": int(
            business_results["false_negatives"]
        ),
        "total_cost": float(
            business_results["total_cost"]
        ),
    }