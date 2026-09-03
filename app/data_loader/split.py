from __future__ import annotations

import pandas as pd


def split_by_time(
    df: pd.DataFrame,
    datetime_column: str = "TX_DATETIME",
    train_end: str = "2018-07-31 23:59:59",
    validation_end: str = "2018-08-31 23:59:59",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split transaction data chronologically.

    Train:
        2018-04-01 through 2018-07-31

    Validation:
        2018-08-01 through 2018-08-31

    Test:
        2018-09-01 through 2018-09-30
    """

    if datetime_column not in df.columns:
        raise ValueError(
            f"Missing datetime column: {datetime_column}"
        )

    data = df.copy()

    data[datetime_column] = pd.to_datetime(
        data[datetime_column]
    )

    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)

    train = data[
        data[datetime_column] <= train_end
    ].copy()

    validation = data[
        (data[datetime_column] > train_end)
        & (data[datetime_column] <= validation_end)
    ].copy()

    test = data[
        data[datetime_column] > validation_end
    ].copy()

    return train, validation, test