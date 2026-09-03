from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    DriftDetectionError,
    FeatureValidationError,
    FraudRiskPlatformError,
    ModelLoadError,
    ModelPredictionError,
    MonitoringError,
    PerformanceEvaluationError,
)
from app.core.logging import get_logger


logger = get_logger(__name__)


def register_exception_handlers(
    app: FastAPI,
) -> None:
    """
    Register centralized exception handlers for
    application-specific and unexpected errors.
    """

    @app.exception_handler(
        FeatureValidationError
    )
    async def feature_validation_error_handler(
        request: Request,
        exc: FeatureValidationError,
    ) -> JSONResponse:
        """
        Handle feature validation failures.
        """

        logger.warning(
            "Feature validation failed | "
            "path=%s | error=%s",
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            content={
                "detail": str(exc),
                "error_type": (
                    "feature_validation_error"
                ),
            },
        )

    @app.exception_handler(
        ModelLoadError
    )
    async def model_load_error_handler(
        request: Request,
        exc: ModelLoadError,
    ) -> JSONResponse:
        """
        Handle production model loading failures.
        """

        logger.error(
            "Model loading failed | "
            "path=%s | error=%s",
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            content={
                "detail": str(exc),
                "error_type": (
                    "model_load_error"
                ),
            },
        )

    @app.exception_handler(
        ModelPredictionError
    )
    async def model_prediction_error_handler(
        request: Request,
        exc: ModelPredictionError,
    ) -> JSONResponse:
        """
        Handle fraud prediction failures.
        """

        logger.error(
            "Prediction failed | "
            "path=%s | error=%s",
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            content={
                "detail": (
                    "Fraud prediction could not be "
                    "completed."
                ),
                "error_type": (
                    "model_prediction_error"
                ),
            },
        )

    @app.exception_handler(
        PerformanceEvaluationError
    )
    async def performance_evaluation_error_handler(
        request: Request,
        exc: PerformanceEvaluationError,
    ) -> JSONResponse:
        """
        Handle model performance evaluation failures.
        """

        logger.warning(
            "Performance evaluation failed | "
            "path=%s | error=%s",
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            content={
                "detail": str(exc),
                "error_type": (
                    "performance_evaluation_error"
                ),
            },
        )

    @app.exception_handler(
        DriftDetectionError
    )
    async def drift_detection_error_handler(
        request: Request,
        exc: DriftDetectionError,
    ) -> JSONResponse:
        """
        Handle drift detection failures.
        """

        logger.warning(
            "Drift detection failed | "
            "path=%s | error=%s",
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            content={
                "detail": str(exc),
                "error_type": (
                    "drift_detection_error"
                ),
            },
        )

    @app.exception_handler(
        MonitoringError
    )
    async def monitoring_error_handler(
        request: Request,
        exc: MonitoringError,
    ) -> JSONResponse:
        """
        Handle general monitoring failures.
        """

        logger.error(
            "Monitoring operation failed | "
            "path=%s | error=%s",
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            content={
                "detail": (
                    "Monitoring operation could not be "
                    "completed."
                ),
                "error_type": (
                    "monitoring_error"
                ),
            },
        )

    @app.exception_handler(
        FraudRiskPlatformError
    )
    async def fraud_risk_platform_error_handler(
        request: Request,
        exc: FraudRiskPlatformError,
    ) -> JSONResponse:
        """
        Handle uncategorized application-specific errors.
        """

        logger.error(
            "Application error | "
            "path=%s | error=%s",
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            content={
                "detail": (
                    "An application error occurred."
                ),
                "error_type": (
                    "fraud_risk_platform_error"
                ),
            },
        )

    @app.exception_handler(
        Exception
    )
    async def unexpected_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """
        Handle unexpected application errors.

        The original error is logged internally while
        the API returns a safe generic response.
        """

        logger.exception(
            "Unexpected error | "
            "path=%s | error=%s",
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            content={
                "detail": (
                    "An unexpected internal server error "
                    "occurred."
                ),
                "error_type": (
                    "internal_server_error"
                ),
            },
        )