from __future__ import annotations

import logging
from pathlib import Path


LOG_DIRECTORY = Path("logs")
LOG_FILE = LOG_DIRECTORY / "fraud_risk_platform.log"


def configure_logging() -> None:
    """
    Configure application-wide logging.

    Logs are written to both the console and a persistent
    application log file.
    """

    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                LOG_FILE,
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def get_logger(
    name: str,
) -> logging.Logger:
    """
    Return a configured application logger.
    """

    return logging.getLogger(name)