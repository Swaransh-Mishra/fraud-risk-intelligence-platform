from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from app.evaluation.metrics import evaluate_model


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
    config: dict,
) -> XGBClassifier:
    """
    Train XGBoost using the configuration
    selected during validation-based tuning.
    """

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

    return model


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate a focused set of CatBoost configurations
    on the validation dataset.
    """

    configurations = [
        {
            "name": "catboost_baseline",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
        },
        {
            "name": "catboost_more_iterations",
            "iterations": 700,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
        },
        {
            "name": "catboost_deeper",
            "iterations": 500,
            "depth": 7,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
        },
        {
            "name": "catboost_shallower",
            "iterations": 700,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
        },
        {
            "name": "catboost_regularized",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 8.0,
        },
    ]

    results = []

    for config in configurations:
        model = CatBoostClassifier(
            iterations=config["iterations"],
            depth=config["depth"],
            learning_rate=config["learning_rate"],
            l2_leaf_reg=config["l2_leaf_reg"],
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
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
                "iterations": config["iterations"],
                "depth": config["depth"],
                "learning_rate": config["learning_rate"],
                "l2_leaf_reg": config["l2_leaf_reg"],
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


def train_tuned_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> CatBoostClassifier:
    """
    Train CatBoost using the configuration
    selected during validation-based tuning.
    """

    model = CatBoostClassifier(
        iterations=config["iterations"],
        depth=config["depth"],
        learning_rate=config["learning_rate"],
        l2_leaf_reg=config["l2_leaf_reg"],
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(X_train, y_train)

    return model