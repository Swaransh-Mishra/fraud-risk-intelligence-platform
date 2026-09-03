from __future__ import annotations

import numpy as np
import pandas as pd

from app.evaluation.metrics import (
    evaluate_model,
    evaluate_probabilities,
)


DEFAULT_THRESHOLD_GRID = np.arange(
    0.05,
    1.00,
    0.05,
)


def analyze_probability_thresholds(
    probabilities: np.ndarray,
    y: pd.Series,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate precomputed probabilities across thresholds."""

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLD_GRID

    results = []

    for threshold in thresholds:
        metrics = evaluate_probabilities(
            probabilities=probabilities,
            y=y,
            threshold=float(threshold),
        )

        results.append(metrics)

    return pd.DataFrame(results)


def analyze_thresholds(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate a model across probability thresholds."""

    probabilities = model.predict_proba(X)[:, 1]

    return analyze_probability_thresholds(
        probabilities=probabilities,
        y=y,
        thresholds=thresholds,
    )