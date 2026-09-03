from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

APP_DIR = PROJECT_ROOT / "app"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

LOGS_DIR = PROJECT_ROOT / "logs"

MODEL_PATH = (
    ARTIFACTS_DIR
    / "fraud_risk_model.joblib"
)

MODEL_METADATA_PATH = (
    ARTIFACTS_DIR
    / "model_metadata.json"
)

EXPERIMENTS_DIR = (
    ARTIFACTS_DIR
    / "experiments"
)


def ensure_directories() -> None:
    """
    Create application directories required during
    runtime if they do not already exist.
    """

    required_directories = [
        ARTIFACTS_DIR,
        DATA_DIR,
        RAW_DATA_DIR,
        LOGS_DIR,
        EXPERIMENTS_DIR,
    ]

    for directory in required_directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )