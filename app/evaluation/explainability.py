from __future__ import annotations

import numpy as np
import pandas as pd


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