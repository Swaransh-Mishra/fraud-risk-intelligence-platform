from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_prediction_errors(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Analyze final fraud model predictions.

    Each transaction is classified into one of four
    prediction outcomes:

    - True Positive
    - False Positive
    - True Negative
    - False Negative

    The returned dataset preserves all model features and adds
    prediction probabilities and error classifications.
    """

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    analysis = X.copy()

    analysis["actual_label"] = (
        np.asarray(y)
    )

    analysis["fraud_probability"] = (
        probabilities
    )

    analysis["predicted_label"] = (
        predictions
    )

    conditions = [
        (
            (analysis["actual_label"] == 1)
            & (analysis["predicted_label"] == 1)
        ),
        (
            (analysis["actual_label"] == 0)
            & (analysis["predicted_label"] == 1)
        ),
        (
            (analysis["actual_label"] == 1)
            & (analysis["predicted_label"] == 0)
        ),
        (
            (analysis["actual_label"] == 0)
            & (analysis["predicted_label"] == 0)
        ),
    ]

    outcomes = [
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]

    analysis["prediction_outcome"] = (
        np.select(
            conditions,
            outcomes,
            default="unknown",
        )
    )

    return analysis


def summarize_prediction_outcomes(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the number and percentage of transactions
    in each prediction outcome category.
    """

    summary = (
        analysis["prediction_outcome"]
        .value_counts()
        .rename_axis("prediction_outcome")
        .reset_index(name="transaction_count")
    )

    summary["percentage"] = (
        summary["transaction_count"]
        / len(analysis)
        * 100
    )

    return summary


def compare_error_groups(
    analysis: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compare feature averages across prediction outcomes.

    This helps identify behavioural differences between
    correctly classified transactions and model errors.
    """

    if features is None:
        features = [
            "TX_AMOUNT",
            "customer_tx_count",
            "customer_amount_ratio",
            "customer_amount_deviation",
            "customer_tx_count_24h",
            "terminal_tx_count",
            "terminal_fraud_count",
            "terminal_fraud_rate",
            "terminal_tx_count_24h",
        ]

    missing_features = (
        set(features)
        - set(analysis.columns)
    )

    if missing_features:
        raise ValueError(
            "Missing analysis features: "
            f"{sorted(missing_features)}"
        )

    comparison = (
        analysis
        .groupby("prediction_outcome")[features]
        .mean()
        .T
        .reset_index()
        .rename(
            columns={
                "index": "feature",
            }
        )
    )

    return comparison


def summarize_false_negatives(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize fraud transactions that the model missed.
    """

    false_negatives = analysis.loc[
        analysis["prediction_outcome"]
        == "false_negative"
    ].copy()

    if false_negatives.empty:
        return pd.DataFrame()

    features = [
        "TX_AMOUNT",
        "fraud_probability",
        "customer_tx_count",
        "customer_amount_ratio",
        "customer_amount_deviation",
        "terminal_tx_count",
        "terminal_fraud_count",
        "terminal_fraud_rate",
    ]

    available_features = [
        feature
        for feature in features
        if feature in false_negatives.columns
    ]

    return (
        false_negatives[available_features]
        .describe()
        .T
    )


def summarize_false_positives(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize legitimate transactions that were
    incorrectly flagged as fraud.
    """

    false_positives = analysis.loc[
        analysis["prediction_outcome"]
        == "false_positive"
    ].copy()

    if false_positives.empty:
        return pd.DataFrame()

    features = [
        "TX_AMOUNT",
        "fraud_probability",
        "customer_tx_count",
        "customer_amount_ratio",
        "customer_amount_deviation",
        "terminal_tx_count",
        "terminal_fraud_count",
        "terminal_fraud_rate",
    ]

    available_features = [
        feature
        for feature in features
        if feature in false_positives.columns
    ]

    return (
        false_positives[available_features]
        .describe()
        .T
    )


def get_high_confidence_errors(
    analysis: pd.DataFrame,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Return the most confident model mistakes.

    False positives are sorted by highest predicted
    fraud probability.

    False negatives are also sorted by highest predicted
    fraud probability, identifying fraudulent transactions
    that received relatively high fraud probabilities but
    were still classified as legitimate at the selected
    threshold.
    """

    false_positives = (
        analysis.loc[
            analysis["prediction_outcome"]
            == "false_positive"
        ]
        .sort_values(
            "fraud_probability",
            ascending=False,
        )
        .head(top_n)
    )

    false_negatives = (
        analysis.loc[
            analysis["prediction_outcome"]
            == "false_negative"
        ]
        .sort_values(
            "fraud_probability",
            ascending=False,
        )
        .head(top_n)
    )

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }