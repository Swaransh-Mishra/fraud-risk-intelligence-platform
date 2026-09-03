from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_performance_endpoint_returns_metrics():
    """
    Verify the performance monitoring endpoint
    returns the expected aggregated metrics.
    """

    response = client.post(
        "/monitoring/performance",
        json={
            "actual_labels": [
                0,
                0,
                1,
                1,
                1,
            ],
            "predicted_labels": [
                0,
                1,
                1,
                1,
                0,
            ],
            "fraud_probabilities": [
                0.1,
                0.6,
                0.9,
                0.8,
                0.2,
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_transactions"] == 5

    assert data["actual_fraud_count"] == 3

    assert data["predicted_fraud_count"] == 3

    assert data["accuracy"] == 0.6

    assert data["precision"] == 0.666667

    assert data["recall"] == 0.666667

    assert data["f1_score"] == 0.666667

    assert data["roc_auc"] == 0.833333


def test_performance_endpoint_without_probabilities():
    """
    Verify the endpoint supports performance
    evaluation without fraud probabilities.
    """

    response = client.post(
        "/monitoring/performance",
        json={
            "actual_labels": [
                0,
                0,
                1,
                1,
            ],
            "predicted_labels": [
                0,
                1,
                1,
                0,
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_transactions"] == 4

    assert data["actual_fraud_count"] == 2

    assert data["predicted_fraud_count"] == 2

    assert data["roc_auc"] is None


def test_performance_endpoint_single_class_returns_none_for_roc_auc():
    """
    Verify ROC-AUC is returned as None when the
    actual labels contain only one class.
    """

    response = client.post(
        "/monitoring/performance",
        json={
            "actual_labels": [
                0,
                0,
                0,
            ],
            "predicted_labels": [
                0,
                0,
                0,
            ],
            "fraud_probabilities": [
                0.1,
                0.2,
                0.3,
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_transactions"] == 3

    assert data["roc_auc"] is None


def test_performance_endpoint_rejects_mismatched_prediction_lengths():
    """
    Verify mismatched actual and predicted label
    lengths are rejected.
    """

    response = client.post(
        "/monitoring/performance",
        json={
            "actual_labels": [
                0,
                1,
                1,
            ],
            "predicted_labels": [
                0,
                1,
            ],
        },
    )

    assert response.status_code == 400


def test_performance_endpoint_rejects_mismatched_probability_lengths():
    """
    Verify mismatched probability lengths are
    rejected.
    """

    response = client.post(
        "/monitoring/performance",
        json={
            "actual_labels": [
                0,
                1,
                1,
            ],
            "predicted_labels": [
                0,
                1,
                1,
            ],
            "fraud_probabilities": [
                0.1,
                0.9,
            ],
        },
    )

    assert response.status_code == 400


def test_performance_endpoint_rejects_empty_datasets():
    """
    Verify empty performance datasets are rejected
    by request validation.
    """

    response = client.post(
        "/monitoring/performance",
        json={
            "actual_labels": [],
            "predicted_labels": [],
        },
    )

    assert response.status_code == 422