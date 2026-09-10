from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.core import (
    MODEL_METADATA_PATH,
    MODEL_PATH,
    FeatureValidationError,
    ModelPredictionError,
    get_logger,
)
from app.features.preprocessing import MODEL_FEATURES
from app.models.model_io import (
    load_model,
    load_model_metadata,
)


logger = get_logger(__name__)

DEFAULT_MODEL_PATH = MODEL_PATH
DEFAULT_METADATA_PATH = MODEL_METADATA_PATH


class FraudRiskPredictor:
    """
    Production inference wrapper for the saved fraud risk model.

    The predictor loads the trained model and its metadata,
    validates the production feature schema, and generates
    fraud predictions.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        self.model = load_model(
            self.model_path
        )

        self.metadata = load_model_metadata(
            self.metadata_path
        )

        if "decision_threshold" not in self.metadata:
            raise ModelPredictionError(
                "Production model metadata does not "
                "contain a decision threshold."
            )

        self.decision_threshold = float(
            self.metadata[
                "decision_threshold"
            ]
        )

        self.required_features = list(
            self.metadata.get(
                "features",
                MODEL_FEATURES,
            )
        )

        if not self.required_features:
            raise ModelPredictionError(
                "Production model metadata does not "
                "contain a valid feature schema."
            )

        logger.info(
            "Fraud risk predictor initialized | "
            "model_name=%s | "
            "model_type=%s | "
            "feature_count=%s | "
            "decision_threshold=%.2f",
            self.metadata.get(
                "model_name",
                "unknown",
            ),
            self.metadata.get(
                "model_type",
                "unknown",
            ),
            len(self.required_features),
            self.decision_threshold,
        )

    def validate_features(
        self,
        X: pd.DataFrame,
    ) -> None:
        """
        Validate that prediction data contains all
        features required by the production model.
        """

        if X.empty:
            raise FeatureValidationError(
                "Prediction input cannot be empty."
            )

        missing_features = (
            set(self.required_features)
            - set(X.columns)
        )

        if missing_features:
            logger.warning(
                "Missing required model features | "
                "missing_features=%s",
                sorted(missing_features),
            )

            raise FeatureValidationError(
                "Missing required model features: "
                f"{sorted(missing_features)}"
            )

    def prepare_features(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and order features according to the
        production model schema.
        """

        self.validate_features(X)

        return X[
            self.required_features
        ].copy()

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate fraud predictions using the saved
        production model and metadata threshold.
        """

        try:
            X_prepared = self.prepare_features(X)

            fraud_probabilities = (
                self.model.predict_proba(
                    X_prepared
                )[:, 1]
            )

            predictions = pd.DataFrame(
                {
                    "fraud_probability": (
                        fraud_probabilities
                    ),
                    "fraud_risk_score": (
                        fraud_probabilities * 100
                    ),
                },
                index=X_prepared.index,
            )

            predictions[
                "predicted_fraud"
            ] = (
                predictions[
                    "fraud_probability"
                ]
                >= self.decision_threshold
            ).astype(int)

            predictions[
                "risk_level"
            ] = (
                predictions[
                    "fraud_probability"
                ].apply(
                    self._get_risk_level
                )
            )

            logger.info(
                "Fraud prediction completed | "
                "total_transactions=%s | "
                "predicted_fraud_count=%s",
                len(predictions),
                int(
                    predictions[
                        "predicted_fraud"
                    ].sum()
                ),
            )

            return predictions

        except FeatureValidationError:
            raise

        except ModelPredictionError:
            raise

        except Exception as error:
            logger.exception(
                "Fraud prediction failed."
            )

            raise ModelPredictionError(
                "Failed to generate fraud predictions."
            ) from error

    def predict_single(
        self,
        transaction_features: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a fraud prediction for one transaction.
        """

        try:
            X = pd.DataFrame(
                [transaction_features]
            )

            predictions = self.predict(X)

            return predictions.iloc[
                0
            ].to_dict()

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
                "Failed to generate single fraud "
                "prediction."
            ) from error

    @staticmethod
    def _get_risk_level(
        fraud_probability: float,
    ) -> str:
        """
        Convert fraud probability into an
        interpretable operational risk level.
        """

        if fraud_probability >= 0.8:
            return "critical"

        if fraud_probability >= 0.6:
            return "high"

        if fraud_probability >= 0.4:
            return "medium"

        if fraud_probability >= 0.2:
            return "low"

        return "minimal"

# ============================================================
# PRODUCTION PREDICTOR INSTANCE
# ============================================================

predictor = FraudRiskPredictor()
