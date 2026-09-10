from pathlib import Path

import pytest

import app.tracking.experiment as experiment_module


@pytest.fixture
def experiments_directory(
    monkeypatch,
    tmp_path,
):
    """
    Redirect experiment storage to an isolated
    temporary directory for testing.
    """

    temporary_directory = (
        tmp_path / "experiments"
    )

    monkeypatch.setattr(
        experiment_module,
        "EXPERIMENTS_DIR",
        temporary_directory,
    )

    return temporary_directory


def test_log_experiment_creates_record(
    experiments_directory,
):
    """
    Verify that an experiment is logged with the
    expected structure.
    """

    result = experiment_module.log_experiment(
        experiment_name=(
            "fraud_model_comparison"
        ),
        model_name="HistGradientBoosting",
        parameters={
            "max_iter": 300,
        },
        metrics={
            "f1_score": 0.72,
            "roc_auc": 0.94,
        },
        metadata={
            "dataset": (
                "credit_card_fraud"
            ),
        },
    )

    assert result["experiment_name"] == (
        "fraud_model_comparison"
    )

    assert result["model_name"] == (
        "HistGradientBoosting"
    )

    assert result["parameters"] == {
        "max_iter": 300,
    }

    assert result["metrics"] == {
        "f1_score": 0.72,
        "roc_auc": 0.94,
    }

    assert "timestamp" in result


def test_log_experiment_creates_storage_file(
    experiments_directory,
):
    """
    Verify that experiment logging creates the
    expected JSONL storage file.
    """

    experiment_module.log_experiment(
        experiment_name="model_test",
        model_name="XGBoost",
        parameters={},
        metrics={},
    )

    expected_file = (
        experiments_directory
        / "model_test.jsonl"
    )

    assert expected_file.exists()


def test_load_experiments_for_missing_file(
    experiments_directory,
):
    """
    Verify that loading a missing experiment returns
    an empty list.
    """

    experiments = (
        experiment_module.load_experiments(
            "missing_experiment"
        )
    )

    assert experiments == []


def test_load_multiple_experiments(
    experiments_directory,
):
    """
    Verify that multiple experiment runs are loaded
    correctly.
    """

    experiment_module.log_experiment(
        experiment_name="fraud_models",
        model_name="HistGradientBoosting",
        parameters={
            "max_iter": 300,
        },
        metrics={
            "f1_score": 0.72,
        },
    )

    experiment_module.log_experiment(
        experiment_name="fraud_models",
        model_name="XGBoost",
        parameters={
            "n_estimators": 500,
        },
        metrics={
            "f1_score": 0.78,
        },
    )

    experiments = (
        experiment_module.load_experiments(
            "fraud_models"
        )
    )

    assert len(experiments) == 2

    assert experiments[0]["model_name"] == (
        "HistGradientBoosting"
    )

    assert experiments[1]["model_name"] == (
        "XGBoost"
    )


def test_best_experiment_higher_metric(
    experiments_directory,
):
    """
    Verify that the experiment with the highest
    metric is selected when higher is better.
    """

    experiment_module.log_experiment(
        experiment_name="model_selection",
        model_name="Model_A",
        parameters={},
        metrics={
            "f1_score": 0.65,
        },
    )

    experiment_module.log_experiment(
        experiment_name="model_selection",
        model_name="Model_B",
        parameters={},
        metrics={
            "f1_score": 0.82,
        },
    )

    best_experiment = (
        experiment_module.get_best_experiment(
            experiment_name=(
                "model_selection"
            ),
            metric_name="f1_score",
            higher_is_better=True,
        )
    )

    assert best_experiment is not None

    assert best_experiment["model_name"] == (
        "Model_B"
    )


def test_best_experiment_lower_metric(
    experiments_directory,
):
    """
    Verify that the experiment with the lowest
    metric is selected when lower is better.
    """

    experiment_module.log_experiment(
        experiment_name="latency_selection",
        model_name="Model_A",
        parameters={},
        metrics={
            "latency_ms": 120,
        },
    )

    experiment_module.log_experiment(
        experiment_name="latency_selection",
        model_name="Model_B",
        parameters={},
        metrics={
            "latency_ms": 85,
        },
    )

    best_experiment = (
        experiment_module.get_best_experiment(
            experiment_name=(
                "latency_selection"
            ),
            metric_name="latency_ms",
            higher_is_better=False,
        )
    )

    assert best_experiment is not None

    assert best_experiment["model_name"] == (
        "Model_B"
    )


def test_best_experiment_missing_metric(
    experiments_directory,
):
    """
    Verify that None is returned when the requested
    metric does not exist.
    """

    experiment_module.log_experiment(
        experiment_name="missing_metric_test",
        model_name="Model_A",
        parameters={},
        metrics={
            "f1_score": 0.75,
        },
    )

    result = (
        experiment_module.get_best_experiment(
            experiment_name=(
                "missing_metric_test"
            ),
            metric_name="roc_auc",
        )
    )

    assert result is None


def test_experiment_names_are_normalized(
    experiments_directory,
):
    """
    Verify that experiment names are converted into
    safe storage filenames.
    """

    experiment_module.log_experiment(
        experiment_name=(
            "Fraud Model Comparison"
        ),
        model_name="XGBoost",
        parameters={},
        metrics={},
    )

    expected_file = (
        experiments_directory
        / "fraud_model_comparison.jsonl"
    )

    assert expected_file.exists()