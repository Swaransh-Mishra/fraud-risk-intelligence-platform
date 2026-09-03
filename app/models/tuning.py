from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from app.evaluation.metrics import evaluate_model


def tune_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate a focused set of HistGradientBoosting configurations
    on the validation dataset.
    """

    configurations = [
        {
            "name": "hgb_baseline",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
        {
            "name": "hgb_more_iterations",
            "learning_rate": 0.03,
            "max_iter": 500,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
        {
            "name": "hgb_deeper",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
        {
            "name": "hgb_regularized",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
        },
        {
            "name": "hgb_conservative",
            "learning_rate": 0.03,
            "max_iter": 500,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
        },
    ]

    results = []

    for config in configurations:
        model = HistGradientBoostingClassifier(
            learning_rate=config["learning_rate"],
            max_iter=config["max_iter"],
            max_leaf_nodes=config["max_leaf_nodes"],
            min_samples_leaf=config["min_samples_leaf"],
            l2_regularization=config["l2_regularization"],
            random_state=42,
        )

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
            threshold=0.5,
        )

        results.append(
            {
                "model": config["name"],
                "learning_rate": config["learning_rate"],
                "max_iter": config["max_iter"],
                "max_leaf_nodes": config["max_leaf_nodes"],
                "min_samples_leaf": config["min_samples_leaf"],
                "l2_regularization": config["l2_regularization"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by="pr_auc", ascending=False)
        .reset_index(drop=True)
    )


def train_tuned_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> HistGradientBoostingClassifier:
    """
    Train the best HistGradientBoosting configuration
    selected from validation-based hyperparameter tuning.
    """

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate a focused set of XGBoost configurations
    on the validation dataset.
    """

    configurations = [
        {
            "name": "xgb_baseline",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "name": "xgb_shallow",
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "name": "xgb_deeper",
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "name": "xgb_regularized",
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "name": "xgb_conservative",
            "n_estimators": 600,
            "max_depth": 3,
            "learning_rate": 0.02,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    ]

    results = []

    for config in configurations:
        model = XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
            threshold=0.5,
        )

        results.append(
            {
                "model": config["name"],
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "learning_rate": config["learning_rate"],
                "subsample": config["subsample"],
                "colsample_bytree": config["colsample_bytree"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by="pr_auc", ascending=False)
        .reset_index(drop=True)
    )


def train_tuned_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """
    Train the best XGBoost configuration selected
    from validation-based hyperparameter tuning.
    """

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model