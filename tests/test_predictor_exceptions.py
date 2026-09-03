from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from app.core.exceptions import (
    FeatureValidationError,
    ModelPredictionError,
)
from app.inference.predictor import (
    FraudRiskPredictor,
)


def test_validate_features_rejects_empty_dataframe():
    """
    Verify that empty prediction input raises a
    FeatureValidationError.
    """

    predictor = FraudRiskPredictor.__new__(
        FraudRiskPredictor
    )

    predictor.required_features = [
        "TX_AMOUNT"
    ]

    empty_data = pd.DataFrame()

    with pytest.raises(
        FeatureValidationError,
        match="Prediction input cannot be empty",
    ):
        predictor.validate_features(
            empty_data
        )


def test_validate_features_rejects_missing_features():
    """
    Verify that missing required model features raise
    a FeatureValidationError.
    """

    predictor = FraudRiskPredictor.__new__(
        FraudRiskPredictor
    )

    predictor.required_features = [
        "TX_AMOUNT",
        "hour_of_day",
    ]

    prediction_data = pd.DataFrame(
        {
            "TX_AMOUNT": [100.0],
        }
    )

    with pytest.raises(
        FeatureValidationError,
        match="Missing required model features",
    ):
        predictor.validate_features(
            prediction_data
        )


def test_prepare_features_orders_columns_correctly():
    """
    Verify that prediction features are returned in
    the exact production model feature order.
    """

    predictor = FraudRiskPredictor.__new__(
        FraudRiskPredictor
    )

    predictor.required_features = [
        "TX_AMOUNT",
        "hour_of_day",
    ]

    prediction_data = pd.DataFrame(
        {
            "hour_of_day": [14],
            "extra_feature": [999],
            "TX_AMOUNT": [250.0],
        }
    )

    prepared_data = predictor.prepare_features(
        prediction_data
    )

    assert list(
        prepared_data.columns
    ) == [
        "TX_AMOUNT",
        "hour_of_day",
    ]


def test_predict_preserves_feature_validation_error():
    """
    Verify that FeatureValidationError is not converted
    into a generic ModelPredictionError.
    """

    predictor = FraudRiskPredictor.__new__(
        FraudRiskPredictor
    )

    predictor.prepare_features = Mock(
        side_effect=FeatureValidationError(
            "Invalid prediction features."
        )
    )

    with pytest.raises(
        FeatureValidationError,
        match="Invalid prediction features",
    ):
        predictor.predict(
            pd.DataFrame(
                {
                    "TX_AMOUNT": [100.0],
                }
            )
        )


def test_predict_wraps_unexpected_error():
    """
    Verify that unexpected prediction failures are
    wrapped as ModelPredictionError.
    """

    predictor = FraudRiskPredictor.__new__(
        FraudRiskPredictor
    )

    predictor.prepare_features = Mock(
        side_effect=RuntimeError(
            "Unexpected preprocessing failure."
        )
    )

    with pytest.raises(
        ModelPredictionError,
        match="Failed to generate fraud predictions",
    ):
        predictor.predict(
            pd.DataFrame(
                {
                    "TX_AMOUNT": [100.0],
                }
            )
        )


def test_predict_single_preserves_known_errors():
    """
    Verify that known prediction exceptions are
    preserved by predict_single.
    """

    predictor = FraudRiskPredictor.__new__(
        FraudRiskPredictor
    )

    predictor.predict = Mock(
        side_effect=ModelPredictionError(
            "Prediction failed."
        )
    )

    with pytest.raises(
        ModelPredictionError,
        match="Prediction failed",
    ):
        predictor.predict_single(
            {
                "TX_AMOUNT": 100.0,
            }
        )


def test_predict_single_wraps_unexpected_error():
    """
    Verify that unexpected single prediction failures
    are wrapped as ModelPredictionError.
    """

    predictor = FraudRiskPredictor.__new__(
        FraudRiskPredictor
    )

    predictor.predict = Mock(
        side_effect=RuntimeError(
            "Unexpected prediction failure."
        )
    )

    with pytest.raises(
        ModelPredictionError,
        match=(
            "Failed to generate single fraud prediction"
        ),
    ):
        predictor.predict_single(
            {
                "TX_AMOUNT": 100.0,
            }
        )


@pytest.mark.parametrize(
    (
        "fraud_probability",
        "expected_risk_level",
    ),
    [
        (0.10, "minimal"),
        (0.20, "low"),
        (0.39, "low"),
        (0.40, "medium"),
        (0.59, "medium"),
        (0.60, "high"),
        (0.79, "high"),
        (0.80, "critical"),
        (0.95, "critical"),
    ],
)
def test_get_risk_level(
    fraud_probability: float,
    expected_risk_level: str,
):
    """
    Verify fraud probability to operational risk level
    mapping.
    """

    risk_level = (
        FraudRiskPredictor._get_risk_level(
            fraud_probability
        )
    )

    assert risk_level == expected_risk_level