from __future__ import annotations

import pandas as pd

from fastapi.testclient import TestClient

from app.core.exceptions import (
    DriftDetectionError,
    FeatureValidationError,
    ModelPredictionError,
    MonitoringError,
    PerformanceEvaluationError,
)
from app.main import app


client = TestClient(
    app,
    raise_server_exceptions=False,
)


VALID_TRANSACTION = {
    "TX_AMOUNT": 49.87,
    "hour_of_day": 0,
    "day_of_week": 5,
    "is_weekend": 1,
    "customer_tx_count": 493,
    "customer_avg_amount": 34.971947,
    "customer_max_amount": 277.15,
    "customer_amount_std": 27.413439,
    "time_since_customer_tx": 18128.0,
    "customer_amount_deviation": 14.898053,
    "customer_amount_ratio": 1.426,
    "customer_tx_count_1h": 0,
    "customer_tx_count_24h": 5,
    "customer_amount_sum_24h": 148.82,
    "terminal_tx_count": 120,
    "terminal_avg_amount": 63.78775,
    "terminal_max_amount": 408.55,
    "terminal_amount_std": 50.07052,
    "terminal_fraud_count": 1,
    "terminal_fraud_rate": 0.008333,
    "terminal_tx_count_1h": 0,
    "terminal_tx_count_24h": 3,
    "terminal_amount_sum_24h": 166.29,
}


def test_predict_endpoint_handles_feature_validation_error(
    monkeypatch,
):
    """
    Verify that a FeatureValidationError raised by the
    production predictor is handled by the centralized
    FastAPI exception handler.
    """

    def raise_feature_validation_error(
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        raise FeatureValidationError(
            "Missing required model features: "
            "['TX_AMOUNT']"
        )

    monkeypatch.setattr(
        "app.main.predictor.predict",
        raise_feature_validation_error,
    )

    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 400

    response_data = response.json()

    assert (
        response_data["error_type"]
        == "feature_validation_error"
    )


def test_predict_endpoint_handles_model_prediction_error(
    monkeypatch,
):
    """
    Verify that a ModelPredictionError raised by the
    production predictor is handled by the centralized
    FastAPI exception handler.
    """

    def raise_model_prediction_error(
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        raise ModelPredictionError(
            "Model prediction failed."
        )

    monkeypatch.setattr(
        "app.main.predictor.predict",
        raise_model_prediction_error,
    )

    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 500

    response_data = response.json()

    assert (
        response_data["error_type"]
        == "model_prediction_error"
    )

    assert (
        response_data["detail"]
        == (
            "Fraud prediction could not be "
            "completed."
        )
    )


def test_monitoring_endpoint_handles_monitoring_error(
    monkeypatch,
):
    """
    Verify that MonitoringError raised by the monitoring
    service is handled by the centralized exception
    handler.
    """

    def raise_monitoring_error() -> dict:
        raise MonitoringError(
            "Monitoring storage failed."
        )

    monkeypatch.setattr(
        "app.main.get_monitoring_report",
        raise_monitoring_error,
    )

    response = client.get(
        "/monitoring"
    )

    assert response.status_code == 500

    response_data = response.json()

    assert (
        response_data["error_type"]
        == "monitoring_error"
    )


def test_drift_endpoint_handles_drift_detection_error(
    monkeypatch,
):
    """
    Verify that DriftDetectionError raised by the drift
    service is handled by the centralized exception
    handler.
    """

    def raise_drift_detection_error(
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: list[str] | None,
        psi_threshold: float,
        ks_pvalue_threshold: float,
    ) -> dict:
        raise DriftDetectionError(
            "Drift calculation failed."
        )

    monkeypatch.setattr(
        "app.main.detect_dataset_drift",
        raise_drift_detection_error,
    )

    response = client.post(
        "/monitoring/drift",
        json={
            "reference_data": [
                {
                    "TX_AMOUNT": 100.0,
                },
                {
                    "TX_AMOUNT": 120.0,
                },
            ],
            "current_data": [
                {
                    "TX_AMOUNT": 200.0,
                },
                {
                    "TX_AMOUNT": 220.0,
                },
            ],
        },
    )

    assert response.status_code == 400

    response_data = response.json()

    assert (
        response_data["error_type"]
        == "drift_detection_error"
    )


def test_performance_endpoint_handles_performance_error(
    monkeypatch,
):
    """
    Verify that PerformanceEvaluationError raised by the
    performance service is handled by the centralized
    exception handler.
    """

    def raise_performance_evaluation_error(
        actual_labels: list[int],
        predicted_labels: list[int],
        fraud_probabilities: list[float] | None,
    ) -> dict:
        raise PerformanceEvaluationError(
            "Performance calculation failed."
        )

    monkeypatch.setattr(
        "app.main.calculate_performance_metrics",
        raise_performance_evaluation_error,
    )

    response = client.post(
        "/monitoring/performance",
        json={
            "actual_labels": [0, 1],
            "predicted_labels": [0, 1],
            "fraud_probabilities": [0.1, 0.9],
        },
    )

    assert response.status_code == 400

    response_data = response.json()

    assert (
        response_data["error_type"]
        == "performance_evaluation_error"
    )


def test_unexpected_api_error_returns_safe_response(
    monkeypatch,
):
    """
    Verify that unexpected API failures do not expose
    internal exception details to API clients.
    """

    def raise_unexpected_error(
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        raise RuntimeError(
            "Sensitive internal implementation details."
        )

    monkeypatch.setattr(
        "app.main.predictor.predict",
        raise_unexpected_error,
    )

    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 500

    response_data = response.json()

    assert (
        response_data["error_type"]
        == "model_prediction_error"
    )

    assert (
        "Sensitive internal implementation details"
        not in response_data["detail"]
    )