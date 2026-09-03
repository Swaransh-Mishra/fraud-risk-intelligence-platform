from __future__ import annotations


class FraudRiskPlatformError(Exception):
    """
    Base exception for all application-specific errors.
    """


class ModelLoadError(FraudRiskPlatformError):
    """
    Raised when the production fraud model
    cannot be loaded.
    """


class ModelPredictionError(FraudRiskPlatformError):
    """
    Raised when fraud-risk prediction fails.
    """


class FeatureValidationError(
    FraudRiskPlatformError,
    ValueError,
):
    """
    Raised when required model features
    are missing or invalid.

    Inherits from ValueError so invalid
    prediction input remains compatible
    with validation and API error-handling
    contracts.
    """


class MonitoringError(FraudRiskPlatformError):
    """
    Raised when monitoring operations fail.
    """


class PerformanceEvaluationError(
    MonitoringError,
    ValueError,
):
    """
    Raised when performance evaluation fails.

    Inherits from ValueError so invalid evaluation
    inputs remain compatible with validation and API
    error-handling contracts.
    """


class DriftDetectionError(
    MonitoringError,
    ValueError,
):
    """
    Raised when drift detection fails.

    Inherits from ValueError so invalid drift
    detection inputs remain compatible with
    validation and API error-handling contracts.
    """