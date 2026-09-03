from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance


def get_xgboost_feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract XGBoost feature importance using the trained booster.
    """

    importance_values = np.asarray(
        model.feature_importances_
    ).reshape(-1)

    if len(feature_names) != len(importance_values):
        raise ValueError(
            "The number of feature names must match "
            "the number of importance values."
        )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    return (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def get_hgb_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "average_precision",
    n_repeats: int = 3,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Calculate permutation importance for HistGradientBoosting.

    Features are ranked by the decrease in the selected
    validation metric when each feature is shuffled.
    """

    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    return (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def get_ensemble_feature_importance(
    hgb_importance: pd.DataFrame,
    xgb_importance: pd.DataFrame,
    hgb_weight: float = 0.7,
    xgb_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Combine normalized feature importance from HGB and XGBoost
    according to the selected ensemble weights.
    """

    if not np.isclose(
        hgb_weight + xgb_weight,
        1.0,
    ):
        raise ValueError(
            "HGB and XGBoost weights must sum to 1.0."
        )

    hgb = hgb_importance[
        ["feature", "importance"]
    ].rename(
        columns={
            "importance": "hgb_importance",
        }
    )

    xgb = xgb_importance[
        ["feature", "importance"]
    ].rename(
        columns={
            "importance": "xgb_importance",
        }
    )

    combined = hgb.merge(
        xgb,
        on="feature",
        how="outer",
    ).fillna(0)

    hgb_total = combined["hgb_importance"].sum()
    xgb_total = combined["xgb_importance"].sum()

    if hgb_total > 0:
        combined["hgb_normalized"] = (
            combined["hgb_importance"]
            / hgb_total
        )
    else:
        combined["hgb_normalized"] = 0.0

    if xgb_total > 0:
        combined["xgb_normalized"] = (
            combined["xgb_importance"]
            / xgb_total
        )
    else:
        combined["xgb_normalized"] = 0.0

    combined["ensemble_importance"] = (
        hgb_weight
        * combined["hgb_normalized"]
        + xgb_weight
        * combined["xgb_normalized"]
    )

    return (
        combined[
            [
                "feature",
                "hgb_importance",
                "xgb_importance",
                "ensemble_importance",
            ]
        ]
        .sort_values(
            "ensemble_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )