from __future__ import annotations
from xgboost import XGBClassifier
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train a class-balanced Random Forest fraud classifier.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def train_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> HistGradientBoostingClassifier:
    """
    Train a gradient boosting fraud classifier.

    The model captures nonlinear relationships between transaction,
    customer, terminal, and temporal features.
    """

    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model

def train_xgboost(
    X_train,
    y_train,
) -> XGBClassifier:
    """
    Train an XGBoost classifier for fraud detection.
    """

    scale_pos_weight = (
        (y_train == 0).sum()
        / (y_train == 1).sum()
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model