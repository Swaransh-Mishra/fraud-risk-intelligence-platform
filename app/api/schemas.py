from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FraudPredictionRequest(BaseModel):
    """
    Request schema containing the final 23 engineered
    features required by the production fraud model.
    """

    TX_AMOUNT: float = Field(
        ...,
        ge=0,
        description="Transaction amount",
    )

    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour when the transaction occurred",
    )

    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of the week",
    )

    is_weekend: int = Field(
        ...,
        ge=0,
        le=1,
        description="Weekend indicator",
    )

    customer_tx_count: int = Field(
        ...,
        ge=0,
    )

    customer_avg_amount: float
    customer_max_amount: float
    customer_amount_std: float
    time_since_customer_tx: float
    customer_amount_deviation: float
    customer_amount_ratio: float

    customer_tx_count_1h: int = Field(
        ...,
        ge=0,
    )

    customer_tx_count_24h: int = Field(
        ...,
        ge=0,
    )

    customer_amount_sum_24h: float = Field(
        ...,
        ge=0,
    )

    terminal_tx_count: int = Field(
        ...,
        ge=0,
    )

    terminal_avg_amount: float
    terminal_max_amount: float
    terminal_amount_std: float

    terminal_fraud_count: int = Field(
        ...,
        ge=0,
    )

    terminal_fraud_rate: float = Field(
        ...,
        ge=0,
        le=1,
    )

    terminal_tx_count_1h: int = Field(
        ...,
        ge=0,
    )

    terminal_tx_count_24h: int = Field(
        ...,
        ge=0,
    )

    terminal_amount_sum_24h: float = Field(
        ...,
        ge=0,
    )


class FraudPredictionResponse(BaseModel):
    """
    Response schema for a single fraud risk prediction.
    """

    fraud_probability: float
    fraud_risk_score: float
    predicted_fraud: int
    risk_level: str
    decision_threshold: float


class BatchPredictionRequest(BaseModel):
    """
    Request schema for multiple fraud transactions.
    """

    transactions: List[
        FraudPredictionRequest
    ] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "List of transactions for batch prediction."
        ),
    )


class BatchPredictionResponse(BaseModel):
    """
    Response schema for multiple fraud predictions.
    """

    total_transactions: int
    predicted_fraud_count: int
    average_fraud_probability: float
    decision_threshold: float
    predictions: List[FraudPredictionResponse]


class DriftDetectionRequest(BaseModel):
    """
    Request schema for dataset drift detection.

    Both datasets must contain records with the same
    feature columns.
    """

    reference_data: List[
        Dict[str, float]
    ] = Field(
        ...,
        min_length=1,
        description=(
            "Reference dataset representing the baseline "
            "feature distribution."
        ),
    )

    current_data: List[
        Dict[str, float]
    ] = Field(
        ...,
        min_length=1,
        description=(
            "Current dataset to compare against the "
            "reference distribution."
        ),
    )

    features: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        description=(
            "Optional list of features to evaluate. "
            "When omitted, all common numeric features "
            "are evaluated."
        ),
    )

    psi_threshold: float = Field(
        default=0.2,
        ge=0,
        description=(
            "PSI threshold used for feature-level "
            "drift detection."
        ),
    )

    ks_pvalue_threshold: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description=(
            "KS test p-value threshold used for "
            "feature-level drift detection."
        ),
    )


class DriftFeatureResult(BaseModel):
    """
    Drift analysis result for one feature.
    """

    psi: float
    ks_statistic: float
    ks_pvalue: float
    drift_detected: bool


class DriftDetectionResponse(BaseModel):
    """
    Response schema for dataset drift detection.
    """

    drift_detected: bool
    drifted_features: List[str]
    evaluated_features: List[str]

    feature_results: Dict[
        str,
        DriftFeatureResult,
    ]


class PerformanceMonitoringRequest(BaseModel):
    """
    Request schema for evaluating model performance
    against observed transaction outcomes.
    """

    actual_labels: List[int] = Field(
        ...,
        min_length=1,
        description=(
            "Observed fraud labels. "
            "Each value must be 0 or 1."
        ),
    )

    predicted_labels: List[int] = Field(
        ...,
        min_length=1,
        description=(
            "Model fraud predictions. "
            "Each value must be 0 or 1."
        ),
    )

    fraud_probabilities: Optional[
        List[float]
    ] = Field(
        default=None,
        description=(
            "Optional predicted fraud probabilities."
        ),
    )


class PerformanceMonitoringResponse(BaseModel):
    """
    Response schema containing aggregated
    fraud model performance metrics.
    """

    total_transactions: int
    actual_fraud_count: int
    predicted_fraud_count: int

    accuracy: float
    precision: float
    recall: float
    f1_score: float

    roc_auc: Optional[float] = None