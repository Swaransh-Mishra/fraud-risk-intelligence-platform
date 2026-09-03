from __future__ import annotations

import numpy as np
import pandas as pd

from app.evaluation.metrics import evaluate_model


class WeightedProbabilityEnsemble:
    """
    Combine fraud probabilities from multiple trained models.

    Each model contributes a configurable weight to the
    final fraud probability.
    """

    def __init__(
        self,
        models: list,
        weights: list[float],
    ) -> None:
        if len(models) != len(weights):
            raise ValueError(
                "The number of models and weights must match."
            )

        if not np.isclose(sum(weights), 1.0):
            raise ValueError(
                "Ensemble weights must sum to 1.0."
            )

        self.models = models
        self.weights = weights

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        probabilities = []

        for model in self.models:
            model_probabilities = model.predict_proba(X)[:, 1]
            probabilities.append(model_probabilities)

        combined_probability = np.average(
            probabilities,
            axis=0,
            weights=self.weights,
        )

        return np.column_stack(
            [
                1 - combined_probability,
                combined_probability,
            ]
        )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]

        return (probabilities >= 0.5).astype(int)


def compare_ensemble_weights(
    models: list,
    weight_configurations: list[tuple[float, float]],
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """
    Compare different HGB and XGBoost probability weights
    using validation data.
    """

    results = []

    for hgb_weight, xgb_weight in weight_configurations:
        ensemble = WeightedProbabilityEnsemble(
            models=models,
            weights=[hgb_weight, xgb_weight],
        )

        metrics = evaluate_model(
            ensemble,
            X_validation,
            y_validation,
            threshold=0.5,
        )

        results.append(
            {
                "hgb_weight": hgb_weight,
                "xgb_weight": xgb_weight,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            by=["pr_auc", "f1_score"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

def train_final_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> WeightedProbabilityEnsemble:
    """
    Train the final fraud detection ensemble using the
    selected 70/30 HGB/XGBoost configuration.
    """

    from app.models.tuning import (
        train_tuned_hist_gradient_boosting,
        train_tuned_xgboost,
    )

    hgb_model = train_tuned_hist_gradient_boosting(
        X_train,
        y_train,
    )

    xgb_model = train_tuned_xgboost(
        X_train,
        y_train,
    )

    final_model = WeightedProbabilityEnsemble(
        models=[hgb_model, xgb_model],
        weights=[0.7, 0.3],
    )

    return final_model