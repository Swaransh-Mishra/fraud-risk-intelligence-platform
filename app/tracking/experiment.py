from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.core.config import EXPERIMENTS_DIR


def _get_experiment_file_path(
    experiment_name: str,
) -> Path:
    """
    Return the JSONL storage path for an experiment.
    """

    safe_experiment_name = (
        experiment_name
        .strip()
        .lower()
        .replace(" ", "_")
    )

    return (
        EXPERIMENTS_DIR
        / f"{safe_experiment_name}.jsonl"
    )


def log_experiment(
    experiment_name: str,
    model_name: str,
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Log one model experiment with its parameters,
    evaluation metrics, and optional metadata.
    """

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    experiment_record = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "parameters": parameters,
        "metrics": metrics,
        "metadata": metadata or {},
    }

    experiment_file_path = (
        _get_experiment_file_path(
            experiment_name
        )
    )

    with experiment_file_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                experiment_record,
                default=str,
            )
            + "\n"
        )

    return experiment_record


def load_experiments(
    experiment_name: str,
) -> list[Dict[str, Any]]:
    """
    Load all recorded runs for an experiment.
    """

    experiment_file_path = (
        _get_experiment_file_path(
            experiment_name
        )
    )

    if not experiment_file_path.exists():
        return []

    experiments = []

    with experiment_file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            experiments.append(
                json.loads(
                    stripped_line
                )
            )

    return experiments


def get_best_experiment(
    experiment_name: str,
    metric_name: str,
    higher_is_better: bool = True,
) -> Dict[str, Any] | None:
    """
    Return the best experiment according to a
    selected evaluation metric.
    """

    experiments = load_experiments(
        experiment_name
    )

    experiments_with_metric = [
        experiment
        for experiment in experiments
        if metric_name
        in experiment["metrics"]
    ]

    if not experiments_with_metric:
        return None

    return sorted(
        experiments_with_metric,
        key=lambda experiment: experiment[
            "metrics"
        ][metric_name],
        reverse=higher_is_better,
    )[0]