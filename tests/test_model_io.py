from pathlib import Path
import json

from app.models.model_io import (
    load_model,
    load_model_metadata,
)


MODEL_PATH = Path(
    "artifacts/fraud_risk_model.joblib"
)

METADATA_PATH = Path(
    "artifacts/model_metadata.json"
)


def test_load_model():
    """
    Verify that the production model artifact
    loads successfully.
    """

    model = load_model(MODEL_PATH)

    assert model is not None

    assert hasattr(
        model,
        "predict",
    )


def test_load_model_metadata():
    """
    Verify that production model metadata
    loads successfully.
    """

    metadata = load_model_metadata(
        METADATA_PATH
    )

    assert isinstance(
        metadata,
        dict,
    )

    assert "model_name" in metadata


def test_model_metadata_features():
    """
    Verify that metadata contains the expected
    production feature schema.
    """

    metadata = load_model_metadata(
        METADATA_PATH
    )

    features = metadata[
        "features"
    ]

    assert isinstance(
        features,
        list,
    )

    assert metadata[
        "feature_count"
    ] == 23

    assert len(
        features
    ) == 23

    assert len(
        features
    ) == metadata[
        "feature_count"
    ]


def test_model_metadata_json_is_valid():
    """
    Verify that loaded metadata can be
    serialized as valid JSON.
    """

    metadata = load_model_metadata(
        METADATA_PATH
    )

    serialized_metadata = json.dumps(
        metadata
    )

    assert serialized_metadata


def test_loaded_model_can_predict():
    """
    Verify that the production model exposes
    prediction interfaces.
    """

    model = load_model(MODEL_PATH)

    assert hasattr(
        model,
        "predict",
    )

    assert hasattr(
        model,
        "predict_proba",
    )