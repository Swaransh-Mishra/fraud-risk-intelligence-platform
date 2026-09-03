from pathlib import Path

import pandas as pd

from app.data_loader.loader import load_transactions
from app.data_loader.split import split_by_time
from app.features.pipeline import build_feature_dataset
from app.features.preprocessing import prepare_model_data
from app.inference.predictor import FraudRiskPredictor


def load_test_features() -> pd.DataFrame:
    """
    Load the untouched test split and prepare the
    final 23-feature model dataset.
    """

    df = load_transactions(
        Path("data/raw/dataset.zip")
    )

    features = build_feature_dataset(df)

    _, _, test_df = split_by_time(features)

    X_test, _ = prepare_model_data(test_df)

    return X_test


def main() -> None:
    """
    Test the production fraud risk predictor.
    """

    print("Loading production predictor...")

    predictor = FraudRiskPredictor()

    print(
        "Model:",
        predictor.metadata["model_name"],
    )

    print(
        "Decision threshold:",
        predictor.decision_threshold,
    )

    print(
        "Required features:",
        len(predictor.required_features),
    )

    print("\nPreparing test transaction...")

    X_test = load_test_features()

    single_transaction = X_test.iloc[[0]].copy()

    print("\nSINGLE TRANSACTION INPUT")

    print(
        single_transaction.to_string(
            index=False
        )
    )

    print("\nRUNNING SINGLE PREDICTION...")

    single_prediction = predictor.predict(
        single_transaction
    )

    print("\nSINGLE PREDICTION RESULT")

    print(
        single_prediction.to_string(
            index=False
        )
    )

    print("\nRUNNING BATCH PREDICTION...")

    batch_transactions = X_test.head(10).copy()

    batch_predictions = predictor.predict(
        batch_transactions
    )

    print("\nBATCH PREDICTION RESULTS")

    print(
        batch_predictions.to_string(
            index=False
        )
    )

    print(
        "\nBatch records predicted:",
        len(batch_predictions),
    )

    print("\nTESTING FEATURE VALIDATION...")

    invalid_transaction = (
        single_transaction
        .drop(
            columns=[
                predictor.required_features[0]
            ]
        )
    )

    try:
        predictor.predict(
            invalid_transaction
        )

    except ValueError as error:
        print(
            "\nFeature validation working correctly."
        )

        print(
            "Validation error:",
            error,
        )

    print(
        "\nPREDICTOR INFERENCE TEST COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()