from __future__ import annotations


import pytest

from app.main import app
from app.monitoring.analytics import get_monitoring_report
from app.monitoring.logging import log_prediction_event
from fastapi.testclient import TestClient


client = TestClient(app)


@pytest.fixture
def monitoring_log_file(monkeypatch, tmp_path):
    """
    Redirect monitoring storage to an isolated
    temporary JSONL file for testing.
    """

    temporary_log_path = (
        tmp_path / "prediction_events.jsonl"
    )

    import app.monitoring.analytics as analytics_module
    import app.monitoring.logging as logging_module

    monkeypatch.setattr(
        logging_module,
        "LOG_FILE_PATH",
        temporary_log_path,
    )

    monkeypatch.setattr(
        analytics_module,
        "LOG_FILE_PATH",
        temporary_log_path,
    )

    return temporary_log_path


def test_empty_monitoring_report(
    monitoring_log_file,
):
    """
    Verify monitoring returns an empty report
    when no prediction events exist.
    """

    report = get_monitoring_report()

    assert report["summary"]["total_events"] == 0
    assert (
        report["summary"]["total_transactions"]
        == 0
    )
    assert (
        report["summary"]["total_predicted_fraud"]
        == 0
    )
    assert (
        report["summary"]["fraud_alert_rate"]
        == 0.0
    )
    assert (
        report["summary"]
        ["average_fraud_probability"]
        == 0.0
    )

    assert report["event_type_summary"] == []
    assert report["model_activity_summary"] == []
    assert report["latest_events"] == []


def test_single_prediction_event_logging(
    monitoring_log_file,
):
    """
    Verify that a single prediction creates
    a monitoring event.
    """

    log_prediction_event(
        event_type="single_prediction",
        model_name=(
            "70/30 HGB-XGBoost Ensemble"
        ),
        decision_threshold=0.4,
        total_transactions=1,
        predicted_fraud_count=0,
        average_fraud_probability=0.25,
    )

    report = get_monitoring_report()

    assert report["summary"]["total_events"] == 1

    assert (
        report["summary"]["total_transactions"]
        == 1
    )

    assert (
        report["summary"]
        ["total_predicted_fraud"]
        == 0
    )

    assert (
        report["summary"]["fraud_alert_rate"]
        == 0.0
    )

    assert (
        report["summary"]
        ["average_fraud_probability"]
        == 0.25
    )

    assert len(
        report["event_type_summary"]
    ) == 1

    assert (
        report["event_type_summary"][0]
        ["event_type"]
        == "single_prediction"
    )


def test_batch_prediction_event_logging(
    monitoring_log_file,
):
    """
    Verify that a batch prediction creates
    a monitoring event with correct metrics.
    """

    log_prediction_event(
        event_type="batch_prediction",
        model_name=(
            "70/30 HGB-XGBoost Ensemble"
        ),
        decision_threshold=0.4,
        total_transactions=10,
        predicted_fraud_count=3,
        average_fraud_probability=0.45,
    )

    report = get_monitoring_report()

    assert report["summary"]["total_events"] == 1

    assert (
        report["summary"]["total_transactions"]
        == 10
    )

    assert (
        report["summary"]
        ["total_predicted_fraud"]
        == 3
    )

    assert (
        report["summary"]["fraud_alert_rate"]
        == 0.3
    )

    assert (
        report["summary"]
        ["average_fraud_probability"]
        == 0.45
    )


def test_monitoring_event_aggregation(
    monitoring_log_file,
):
    """
    Verify multiple prediction events are
    aggregated correctly.
    """

    log_prediction_event(
        event_type="single_prediction",
        model_name=(
            "70/30 HGB-XGBoost Ensemble"
        ),
        decision_threshold=0.4,
        total_transactions=1,
        predicted_fraud_count=0,
        average_fraud_probability=0.2,
    )

    log_prediction_event(
        event_type="batch_prediction",
        model_name=(
            "70/30 HGB-XGBoost Ensemble"
        ),
        decision_threshold=0.4,
        total_transactions=4,
        predicted_fraud_count=2,
        average_fraud_probability=0.6,
    )

    report = get_monitoring_report()

    assert report["summary"]["total_events"] == 2

    assert (
        report["summary"]["total_transactions"]
        == 5
    )

    assert (
        report["summary"]
        ["total_predicted_fraud"]
        == 2
    )

    assert (
        report["summary"]["fraud_alert_rate"]
        == 0.4
    )

    assert len(
        report["event_type_summary"]
    ) == 2

    assert len(
        report["model_activity_summary"]
    ) == 1

    assert (
        report["model_activity_summary"][0]
        ["event_count"]
        == 2
    )


def test_monitoring_endpoint(
    monitoring_log_file,
):
    """
    Verify the monitoring API endpoint
    returns a successful response.
    """

    response = client.get(
        "/monitoring"
    )

    assert response.status_code == 200

    data = response.json()

    assert "summary" in data
    assert "event_type_summary" in data
    assert "model_activity_summary" in data
    assert "latest_events" in data


def test_latest_events_order(
    monitoring_log_file,
):
    """
    Verify monitoring returns the latest
    events in reverse chronological order.
    """

    log_prediction_event(
        event_type="first_event",
        model_name=(
            "70/30 HGB-XGBoost Ensemble"
        ),
        decision_threshold=0.4,
        total_transactions=1,
        predicted_fraud_count=0,
        average_fraud_probability=0.1,
    )

    log_prediction_event(
        event_type="second_event",
        model_name=(
            "70/30 HGB-XGBoost Ensemble"
        ),
        decision_threshold=0.4,
        total_transactions=1,
        predicted_fraud_count=1,
        average_fraud_probability=0.9,
    )

    report = get_monitoring_report()

    latest_events = report["latest_events"]

    assert len(latest_events) == 2

    assert (
        latest_events[0]["event_type"]
        == "second_event"
    )

    assert (
        latest_events[1]["event_type"]
        == "first_event"
    )