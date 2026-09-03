import pytest

from app.monitoring.performance import (
    calculate_performance_metrics,
)


def test_performance_metrics_for_predictions():
    result = calculate_performance_metrics(
        actual_labels=[0, 0, 1, 1, 1],
        predicted_labels=[0, 1, 1, 1, 0],
        fraud_probabilities=[
            0.1,
            0.6,
            0.9,
            0.8,
            0.2,
        ],
    )

    assert result["total_transactions"] == 5
    assert result["actual_fraud_count"] == 3
    assert result["predicted_fraud_count"] == 3

    assert result["accuracy"] == 0.6
    assert result["precision"] == pytest.approx(
        0.666667,
        abs=1e-6,
    )
    assert result["recall"] == pytest.approx(
        0.666667,
        abs=1e-6,
    )
    assert result["f1_score"] == pytest.approx(
        0.666667,
        abs=1e-6,
    )
    assert result["roc_auc"] == pytest.approx(
        0.833333,
        abs=1e-6,
    )


def test_performance_metrics_without_probabilities():
    result = calculate_performance_metrics(
        actual_labels=[0, 1, 1],
        predicted_labels=[0, 1, 0],
    )

    assert result["total_transactions"] == 3
    assert "roc_auc" not in result


def test_single_class_returns_none_for_roc_auc():
    result = calculate_performance_metrics(
        actual_labels=[0, 0, 0],
        predicted_labels=[0, 0, 0],
        fraud_probabilities=[0.1, 0.2, 0.3],
    )

    assert result["accuracy"] == 1.0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0
    assert result["roc_auc"] is None


def test_empty_actual_labels_raises_error():
    with pytest.raises(
        ValueError,
        match="actual_labels cannot be empty",
    ):
        calculate_performance_metrics(
            actual_labels=[],
            predicted_labels=[],
        )


def test_mismatched_prediction_lengths_raise_error():
    with pytest.raises(
        ValueError,
        match="actual_labels and predicted_labels",
    ):
        calculate_performance_metrics(
            actual_labels=[0, 1],
            predicted_labels=[0],
        )


def test_mismatched_probability_lengths_raise_error():
    with pytest.raises(
        ValueError,
        match="actual_labels and fraud_probabilities",
    ):
        calculate_performance_metrics(
            actual_labels=[0, 1],
            predicted_labels=[0, 1],
            fraud_probabilities=[0.1],
        )