from __future__ import annotations

import numpy as np
import pandas as pd

from app.evaluation.thresholds import DEFAULT_THRESHOLD_GRID


def calculate_business_cost(
    y_true: pd.Series,
    y_pred: np.ndarray,
    false_positive_cost: float = 5.0,
    false_negative_cost: float = 100.0,
) -> dict[str, float]:
    """Calculate estimated fraud business cost."""

    y_true_array = np.asarray(y_true)

    false_positives = (
        (y_true_array == 0) & (y_pred == 1)
    ).sum()

    false_negatives = (
        (y_true_array == 1) & (y_pred == 0)
    ).sum()

    total_cost = (
        false_positives * false_positive_cost
        + false_negatives * false_negative_cost
    )

    return {
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "total_cost": float(total_cost),
    }


def analyze_probability_business_thresholds(
    probabilities: np.ndarray,
    y: pd.Series,
    thresholds: np.ndarray | None = None,
    false_positive_cost: float = 5.0,
    false_negative_cost: float = 100.0,
) -> pd.DataFrame:
    """Evaluate probability thresholds using business cost."""

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLD_GRID

    probabilities = np.asarray(probabilities)

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        cost_results = calculate_business_cost(
            y_true=y,
            y_pred=predictions,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )

        results.append(
            {
                "threshold": float(threshold),
                **cost_results,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("total_cost")
        .reset_index(drop=True)
    )


def analyze_business_thresholds(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: np.ndarray | None = None,
    false_positive_cost: float = 5.0,
    false_negative_cost: float = 100.0,
) -> pd.DataFrame:
    """Evaluate a model using business-cost thresholds."""

    probabilities = model.predict_proba(X)[:, 1]

    return analyze_probability_business_thresholds(
        probabilities=probabilities,
        y=y,
        thresholds=thresholds,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )


def analyze_cost_scenarios(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cost_scenarios: list[dict[str, float]] | None = None,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare optimal thresholds across business scenarios."""

    if cost_scenarios is None:
        cost_scenarios = [
            {
                "scenario": "high_fraud_loss",
                "false_positive_cost": 5.0,
                "false_negative_cost": 100.0,
            },
            {
                "scenario": "balanced_operations",
                "false_positive_cost": 20.0,
                "false_negative_cost": 100.0,
            },
            {
                "scenario": "high_customer_friction",
                "false_positive_cost": 50.0,
                "false_negative_cost": 100.0,
            },
        ]

    results = []

    for scenario in cost_scenarios:
        threshold_results = analyze_business_thresholds(
            model=model,
            X=X,
            y=y,
            thresholds=thresholds,
            false_positive_cost=scenario[
                "false_positive_cost"
            ],
            false_negative_cost=scenario[
                "false_negative_cost"
            ],
        )

        best_result = threshold_results.iloc[0]

        results.append(
            {
                "scenario": scenario["scenario"],
                "false_positive_cost": scenario[
                    "false_positive_cost"
                ],
                "false_negative_cost": scenario[
                    "false_negative_cost"
                ],
                "best_threshold": best_result[
                    "threshold"
                ],
                "false_positives": best_result[
                    "false_positives"
                ],
                "false_negatives": best_result[
                    "false_negatives"
                ],
                "lowest_total_cost": best_result[
                    "total_cost"
                ],
            }
        )

    return pd.DataFrame(results)