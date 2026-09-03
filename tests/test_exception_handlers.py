from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import (
    DriftDetectionError,
    FeatureValidationError,
    FraudRiskPlatformError,
    ModelLoadError,
    ModelPredictionError,
    MonitoringError,
    PerformanceEvaluationError,
)
from app.core.handlers import (
    register_exception_handlers,
)


def create_test_app() -> FastAPI:
    """
    Create an isolated FastAPI application for
    centralized exception-handler testing.
    """

    app = FastAPI()

    register_exception_handlers(app)

    @app.get("/feature-validation-error")
    def raise_feature_validation_error() -> None:
        raise FeatureValidationError(
            "Invalid transaction feature."
        )

    @app.get("/model-load-error")
    def raise_model_load_error() -> None:
        raise ModelLoadError(
            "Production model could not be loaded."
        )

    @app.get("/model-prediction-error")
    def raise_model_prediction_error() -> None:
        raise ModelPredictionError(
            "Fraud prediction failed."
        )

    @app.get("/performance-error")
    def raise_performance_error() -> None:
        raise PerformanceEvaluationError(
            "Performance evaluation failed."
        )

    @app.get("/drift-error")
    def raise_drift_error() -> None:
        raise DriftDetectionError(
            "Dataset drift detection failed."
        )

    @app.get("/monitoring-error")
    def raise_monitoring_error() -> None:
        raise MonitoringError(
            "Monitoring operation failed."
        )

    @app.get("/application-error")
    def raise_application_error() -> None:
        raise FraudRiskPlatformError(
            "General application error."
        )

    @app.get("/unexpected-error")
    def raise_unexpected_error() -> None:
        raise RuntimeError(
            "Unexpected internal failure."
        )

    return app


app = create_test_app()

client = TestClient(
    app,
    raise_server_exceptions=False,
)


def test_feature_validation_error_handler():
    """
    Verify that feature validation failures return
    HTTP 400 with the expected error structure.
    """

    response = client.get(
        "/feature-validation-error"
    )

    body = response.json()

    assert response.status_code == 400

    assert body["detail"] == (
        "Invalid transaction feature."
    )

    assert body["error_type"] == (
        "feature_validation_error"
    )


def test_model_load_error_handler():
    """
    Verify that model loading failures return
    HTTP 503 with the expected error structure.
    """

    response = client.get(
        "/model-load-error"
    )

    body = response.json()

    assert response.status_code == 503

    assert body["detail"] == (
        "Production model could not be loaded."
    )

    assert body["error_type"] == (
        "model_load_error"
    )


def test_model_prediction_error_handler():
    """
    Verify that prediction failures return a safe
    HTTP 500 response without exposing internals.
    """

    response = client.get(
        "/model-prediction-error"
    )

    body = response.json()

    assert response.status_code == 500

    assert body["detail"] == (
        "Fraud prediction could not be completed."
    )

    assert body["error_type"] == (
        "model_prediction_error"
    )


def test_performance_evaluation_error_handler():
    """
    Verify that performance evaluation failures
    return HTTP 400.
    """

    response = client.get(
        "/performance-error"
    )

    body = response.json()

    assert response.status_code == 400

    assert body["detail"] == (
        "Performance evaluation failed."
    )

    assert body["error_type"] == (
        "performance_evaluation_error"
    )


def test_drift_detection_error_handler():
    """
    Verify that drift detection failures return
    HTTP 400.
    """

    response = client.get(
        "/drift-error"
    )

    body = response.json()

    assert response.status_code == 400

    assert body["detail"] == (
        "Dataset drift detection failed."
    )

    assert body["error_type"] == (
        "drift_detection_error"
    )


def test_monitoring_error_handler():
    """
    Verify that monitoring failures return a safe
    HTTP 500 response.
    """

    response = client.get(
        "/monitoring-error"
    )

    body = response.json()

    assert response.status_code == 500

    assert body["detail"] == (
        "Monitoring operation could not be completed."
    )

    assert body["error_type"] == (
        "monitoring_error"
    )


def test_application_error_handler():
    """
    Verify that uncategorized application errors
    return a safe HTTP 500 response.
    """

    response = client.get(
        "/application-error"
    )

    body = response.json()

    assert response.status_code == 500

    assert body["detail"] == (
        "An application error occurred."
    )

    assert body["error_type"] == (
        "fraud_risk_platform_error"
    )


def test_unexpected_error_handler():
    """
    Verify that unexpected exceptions return a safe
    generic HTTP 500 response.
    """

    response = client.get(
        "/unexpected-error"
    )

    body = response.json()

    assert response.status_code == 500

    assert body["detail"] == (
        "An unexpected internal server error "
        "occurred."
    )

    assert body["error_type"] == (
        "internal_server_error"
    )