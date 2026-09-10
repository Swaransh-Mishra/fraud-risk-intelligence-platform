from __future__ import annotations

import pandas as pd
from fastapi import FastAPI

from app.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftDetectionRequest,
    DriftDetectionResponse,
    DriftFeatureResult,
    FraudPredictionRequest,
    FraudPredictionResponse,
    PerformanceMonitoringRequest,
    PerformanceMonitoringResponse,
)
from app.core import (
    configure_logging,
    ensure_directories,
    get_logger,
)
from app.core.exceptions import (
    DriftDetectionError,
    FeatureValidationError,
    ModelPredictionError,
    MonitoringError,
    PerformanceEvaluationError,
)
from app.core.handlers import (
    register_exception_handlers,
)
from app.inference.predictor import FraudRiskPredictor
from app.monitoring.analytics import get_monitoring_report
from app.monitoring.drift import detect_dataset_drift
from app.monitoring.logging import log_prediction_event
from app.monitoring.performance import (
    calculate_performance_metrics,
)


ensure_directories()
configure_logging()

logger = get_logger(__name__)


app = FastAPI(
    title="Fraud Risk Intelligence Platform API",
    description=(
        "Production API for transaction fraud risk "
        "prediction using the finalized production model."
    ),
    version="1.0.0",
)


register_exception_handlers(app)


try:
    predictor = FraudRiskPredictor()

    logger.info(
        "Production fraud risk model loaded successfully."
    )

except Exception as error:
    logger.exception(
        "Failed to initialize the production model."
    )

    raise RuntimeError(
        "Application startup failed because the "
        "production fraud model could not be loaded."
    ) from error


@app.get(
    "/",
    summary="Root",
    description="Basic API status endpoint.",
)
def root() -> dict[str, str]:
    """
    Return the API service status.
    """

    return {
        "message": (
            "Fraud Risk Intelligence Platform API is running"
        )
    }


@app.get(
    "/health",
    summary="Health Check",
    description="Check whether the API service is running.",
)
def health_check() -> dict[str, str]:
    """
    Return the API health status.
    """

    return {
        "status": "healthy"
    }


@app.get(
    "/model-info",
    summary="Model Info",
    description="Return metadata for the production fraud model.",
)
def model_info() -> dict:
    """
    Return production model metadata.
    """

    return predictor.metadata


@app.post(
    "/predict",
    response_model=FraudPredictionResponse,
    summary="Predict Fraud Risk",
    description=(
        "Generate a fraud risk prediction for one "
        "transaction using the production model."
    ),
)
def predict_fraud(
    transaction: FraudPredictionRequest,
) -> FraudPredictionResponse:
    """
    Generate a fraud prediction for one transaction.
    """

    try:
        transaction_data = pd.DataFrame(
            [transaction.model_dump()]
        )

        prediction = predictor.predict(
            transaction_data
        )

        result = prediction.iloc[0].to_dict()

        fraud_probability = round(
            float(
                result["fraud_probability"]
            ),
            6,
        )

        fraud_risk_score = round(
            float(
                result["fraud_risk_score"]
            ),
            2,
        )

        predicted_fraud = int(
            result["predicted_fraud"]
        )

        risk_level = str(
            result["risk_level"]
        )

        log_prediction_event(
            event_type="single_prediction",
            model_name=(
                predictor.metadata["model_name"]
            ),
            decision_threshold=(
                predictor.decision_threshold
            ),
            total_transactions=1,
            predicted_fraud_count=(
                predicted_fraud
            ),
            average_fraud_probability=(
                fraud_probability
            ),
        )

        logger.info(
            "Single fraud prediction completed | "
            "predicted_fraud=%s | "
            "risk_level=%s | "
            "fraud_probability=%.6f",
            predicted_fraud,
            risk_level,
            fraud_probability,
        )

        return FraudPredictionResponse(
            fraud_probability=(
                fraud_probability
            ),
            fraud_risk_score=(
                fraud_risk_score
            ),
            predicted_fraud=(
                predicted_fraud
            ),
            risk_level=risk_level,
            decision_threshold=(
                predictor.decision_threshold
            ),
        )

    except (
        FeatureValidationError,
        ModelPredictionError,
    ):
        raise

    except Exception as error:
        logger.exception(
            "Single fraud prediction failed."
        )

        raise ModelPredictionError(
            "Single fraud prediction failed."
        ) from error


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Predict Fraud Risk",
    description=(
        "Generate fraud risk predictions for multiple "
        "transactions using the production model."
    ),
)
def predict_fraud_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """
    Generate fraud predictions for multiple transactions.
    """

    try:
        transactions_data = pd.DataFrame(
            [
                transaction.model_dump()
                for transaction in request.transactions
            ]
        )

        predictions = predictor.predict(
            transactions_data
        )

        prediction_results = []

        for _, row in predictions.iterrows():
            prediction_results.append(
                FraudPredictionResponse(
                    fraud_probability=round(
                        float(
                            row[
                                "fraud_probability"
                            ]
                        ),
                        6,
                    ),
                    fraud_risk_score=round(
                        float(
                            row[
                                "fraud_risk_score"
                            ]
                        ),
                        2,
                    ),
                    predicted_fraud=int(
                        row[
                            "predicted_fraud"
                        ]
                    ),
                    risk_level=str(
                        row[
                            "risk_level"
                        ]
                    ),
                    decision_threshold=(
                        predictor.decision_threshold
                    ),
                )
            )

        predicted_fraud_count = sum(
            result.predicted_fraud
            for result in prediction_results
        )

        average_fraud_probability = round(
            float(
                predictions[
                    "fraud_probability"
                ].mean()
            ),
            6,
        )

        log_prediction_event(
            event_type="batch_prediction",
            model_name=(
                predictor.metadata["model_name"]
            ),
            decision_threshold=(
                predictor.decision_threshold
            ),
            total_transactions=(
                len(prediction_results)
            ),
            predicted_fraud_count=(
                predicted_fraud_count
            ),
            average_fraud_probability=(
                average_fraud_probability
            ),
        )

        logger.info(
            "Batch fraud prediction completed | "
            "total_transactions=%s | "
            "predicted_fraud_count=%s | "
            "average_fraud_probability=%.6f",
            len(prediction_results),
            predicted_fraud_count,
            average_fraud_probability,
        )

        return BatchPredictionResponse(
            total_transactions=(
                len(prediction_results)
            ),
            predicted_fraud_count=(
                predicted_fraud_count
            ),
            average_fraud_probability=(
                average_fraud_probability
            ),
            decision_threshold=(
                predictor.decision_threshold
            ),
            predictions=prediction_results,
        )

    except (
        FeatureValidationError,
        ModelPredictionError,
    ):
        raise

    except Exception as error:
        logger.exception(
            "Batch fraud prediction failed."
        )

        raise ModelPredictionError(
            "Batch fraud prediction failed."
        ) from error


@app.get(
    "/monitoring",
    summary="Monitoring Report",
    description=(
        "Return production prediction activity "
        "and aggregated monitoring analytics."
    ),
)
def monitoring_report() -> dict:
    """
    Return the production monitoring report.
    """

    try:
        report = get_monitoring_report()

        logger.info(
            "Monitoring report generated successfully."
        )

        return report

    except MonitoringError:
        raise

    except Exception as error:
        logger.exception(
            "Monitoring report generation failed."
        )

        raise MonitoringError(
            "Monitoring report generation failed."
        ) from error


@app.post(
    "/monitoring/drift",
    response_model=DriftDetectionResponse,
    summary="Detect Dataset Drift",
    description=(
        "Compare a current dataset against a reference "
        "dataset and detect feature-level distribution drift "
        "using Population Stability Index and the "
        "Kolmogorov-Smirnov test."
    ),
)
def detect_drift(
    request: DriftDetectionRequest,
) -> DriftDetectionResponse:
    """
    Detect distribution drift between reference and
    current transaction datasets.
    """

    try:
        reference_data = pd.DataFrame(
            request.reference_data
        )

        current_data = pd.DataFrame(
            request.current_data
        )

        report = detect_dataset_drift(
            reference_data=reference_data,
            current_data=current_data,
            features=request.features,
            psi_threshold=request.psi_threshold,
            ks_pvalue_threshold=(
                request.ks_pvalue_threshold
            ),
        )

        feature_results = {
            feature_name: DriftFeatureResult(
                psi=float(
                    result["psi"]
                ),
                ks_statistic=float(
                    result["ks_statistic"]
                ),
                ks_pvalue=float(
                    result["ks_pvalue"]
                ),
                drift_detected=bool(
                    result["drift_detected"]
                ),
            )
            for feature_name, result
            in report[
                "feature_results"
            ].items()
        }

        logger.info(
            "Drift detection completed | "
            "drift_detected=%s | "
            "drifted_feature_count=%s",
            report["drift_detected"],
            len(
                report[
                    "drifted_features"
                ]
            ),
        )

        return DriftDetectionResponse(
            drift_detected=bool(
                report["drift_detected"]
            ),
            drifted_features=list(
                report[
                    "drifted_features"
                ]
            ),
            evaluated_features=list(
                report[
                    "feature_results"
                ].keys()
            ),
            feature_results=feature_results,
        )

    except DriftDetectionError:
        raise

    except ValueError as error:
        logger.warning(
            "Invalid drift detection request: %s",
            str(error),
        )

        raise DriftDetectionError(
            str(error)
        ) from error

    except Exception as error:
        logger.exception(
            "Drift detection failed."
        )

        raise DriftDetectionError(
            "Dataset drift detection failed."
        ) from error


@app.post(
    "/monitoring/performance",
    response_model=PerformanceMonitoringResponse,
    summary="Evaluate Model Performance",
    description=(
        "Calculate fraud model performance metrics using "
        "observed transaction outcomes and model predictions."
    ),
)
def evaluate_model_performance(
    request: PerformanceMonitoringRequest,
) -> PerformanceMonitoringResponse:
    """
    Evaluate fraud model performance against
    observed transaction outcomes.
    """

    try:
        result = calculate_performance_metrics(
            actual_labels=request.actual_labels,
            predicted_labels=request.predicted_labels,
            fraud_probabilities=(
                request.fraud_probabilities
            ),
        )

        roc_auc = result.get(
            "roc_auc"
        )

        logger.info(
            "Performance evaluation completed | "
            "total_transactions=%s | "
            "accuracy=%.6f | "
            "f1_score=%.6f",
            result["total_transactions"],
            result["accuracy"],
            result["f1_score"],
        )

        return PerformanceMonitoringResponse(
            total_transactions=int(
                result[
                    "total_transactions"
                ]
            ),
            actual_fraud_count=int(
                result[
                    "actual_fraud_count"
                ]
            ),
            predicted_fraud_count=int(
                result[
                    "predicted_fraud_count"
                ]
            ),
            accuracy=float(
                result["accuracy"]
            ),
            precision=float(
                result["precision"]
            ),
            recall=float(
                result["recall"]
            ),
            f1_score=float(
                result["f1_score"]
            ),
            roc_auc=(
                float(roc_auc)
                if roc_auc is not None
                else None
            ),
        )

    except PerformanceEvaluationError:
        raise

    except ValueError as error:
        logger.warning(
            "Invalid performance evaluation request: %s",
            str(error),
        )

        raise PerformanceEvaluationError(
            str(error)
        ) from error

    except Exception as error:
        logger.exception(
            "Performance evaluation failed."
        )

        raise PerformanceEvaluationError(
            "Model performance evaluation failed."
        ) from error