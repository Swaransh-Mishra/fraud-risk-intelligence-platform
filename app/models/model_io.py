from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from app.core import (
    ARTIFACTS_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    ModelLoadError,
    get_logger,
)


logger = get_logger(__name__)


DEFAULT_MODEL_DIR = ARTIFACTS_DIR


def save_model(
    model: Any,
    metadata: dict[str, Any],
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    model_filename: str = MODEL_PATH.name,
    metadata_filename: str = MODEL_METADATA_PATH.name,
) -> dict[str, Path]:
    """
    Save a trained fraud detection model and its metadata.

    Parameters
    ----------
    model:
        Trained model object to persist.

    metadata:
        Dictionary containing model configuration and
        production metadata.

    model_dir:
        Directory where artifacts will be stored.

    model_filename:
        Filename for the serialized model.

    metadata_filename:
        Filename for the JSON metadata file.

    Returns
    -------
    dict[str, Path]
        Paths of the saved model and metadata files.
    """

    model_dir = Path(model_dir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_dir
        / model_filename
    )

    metadata_path = (
        model_dir
        / metadata_filename
    )

    try:
        joblib.dump(
            model,
            model_path,
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
            )

        logger.info(
            "Model artifacts saved successfully | "
            "model_path=%s | metadata_path=%s",
            model_path,
            metadata_path,
        )

        return {
            "model_path": model_path,
            "metadata_path": metadata_path,
        }

    except Exception as error:
        logger.exception(
            "Failed to save model artifacts."
        )

        raise ModelLoadError(
            "Failed to save model artifacts."
        ) from error


def load_model(
    model_path: Path | str,
) -> Any:
    """
    Load a previously saved fraud detection model.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise ModelLoadError(
            f"Model file not found: {model_path}"
        )

    try:
        model = joblib.load(
            model_path
        )

        logger.info(
            "Model loaded successfully | "
            "model_path=%s",
            model_path,
        )

        return model

    except ModelLoadError:
        raise

    except Exception as error:
        logger.exception(
            "Failed to load model | "
            "model_path=%s",
            model_path,
        )

        raise ModelLoadError(
            f"Failed to load model: {model_path}"
        ) from error


def load_model_metadata(
    metadata_path: Path | str,
) -> dict[str, Any]:
    """
    Load saved model metadata.
    """

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise ModelLoadError(
            "Metadata file not found: "
            f"{metadata_path}"
        )

    try:
        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        logger.info(
            "Model metadata loaded successfully | "
            "metadata_path=%s",
            metadata_path,
        )

        return metadata

    except ModelLoadError:
        raise

    except Exception as error:
        logger.exception(
            "Failed to load model metadata | "
            "metadata_path=%s",
            metadata_path,
        )

        raise ModelLoadError(
            "Failed to load model metadata: "
            f"{metadata_path}"
        ) from error