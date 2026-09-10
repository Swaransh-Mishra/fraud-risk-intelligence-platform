from __future__ import annotations

import numpy as np
import pandas as pd


class WeightedProbabilityEnsemble:
    """
    Generic probability-weighted ensemble for binary classifiers.

    Each component model must implement predict_proba().
    The ensemble combines positive-class probabilities using
    the supplied weights.
    """

    def __init__(
        self,
        models: list,
        weights: list[float],
    ) -> None:
        if len(models) != len(weights):
            raise ValueError(
                "Number of models must match number of weights."
            )

        if not models:
            raise ValueError(
                "At least one model is required."
            )

        if not np.isclose(sum(weights), 1.0):
            raise ValueError(
                "Ensemble weights must sum to 1.0."
            )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Ensemble weights must be non-negative."
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

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Return the weighted average of normalized component-model
        feature importances.
        """
        importances = []

        for model in self.models:
            if not hasattr(model, "feature_importances_"):
                raise AttributeError(
                    "All ensemble models must expose "
                    "feature_importances_."
                )

            model_importances = np.asarray(
                model.feature_importances_,
                dtype=float,
            )

            total_importance = model_importances.sum()

            if total_importance <= 0:
                raise ValueError(
                    "Model feature importances must have a positive sum."
                )

            importances.append(
                model_importances / total_importance
            )

        return np.average(
            importances,
            axis=0,
            weights=self.weights,
        )