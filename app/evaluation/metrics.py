from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilities(
    probabilities: np.ndarray,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Evaluate binary predictions from probability scores."""

    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
    ).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, predictions),
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(
            y,
            probabilities,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Evaluate a trained binary classification model."""

    probabilities = model.predict_proba(X)[:, 1]

    return evaluate_probabilities(
        probabilities=probabilities,
        y=y,
        threshold=threshold,
    )