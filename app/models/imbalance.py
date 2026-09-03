from __future__ import annotations
from app.evaluation.metrics import evaluate_model
import pandas as pd
from xgboost import XGBClassifier


def tune_xgboost_class_weight(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """
    Compare moderate positive-class weights for XGBoost.

    The validation set is used to identify the best
    class-imbalance trade-off.
    """

    weights = [1, 5, 10, 20, 50]

    results = []

    for weight in weights:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
            threshold=0.5,
        )

        results.append(
            {
                "scale_pos_weight": weight,
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
            by="pr_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )