from __future__ import annotations

import pandas as pd

from app.features.customer import add_customer_features
from app.features.temporal import add_temporal_features
from app.features.terminal import add_terminal_features


def build_feature_dataset(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete fraud-risk feature dataset.

    Features are created using:
    - Temporal transaction information
    - Historical customer behaviour
    - Historical terminal behaviour

    Each feature module uses only information available before
    the current transaction where historical behaviour is involved.
    """

    features = transactions.copy()

    # Transaction time features
    features = add_temporal_features(features)

    # Customer behaviour features
    features = add_customer_features(features)

    # Terminal behaviour features
    features = add_terminal_features(features)

    return features