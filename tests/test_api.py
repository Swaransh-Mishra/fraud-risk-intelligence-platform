from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.inference.predictor import predictor

client = TestClient(app)

VALID_TRANSACTION = {
    "TX_AMOUNT": 49.87,
    "hour_of_day": 0,
    "day_of_week": 5,
    "is_weekend": 1,
    "customer_tx_count": 493,
    "customer_avg_amount": 34.971947,
    "customer_max_amount": 277.15,
    "customer_amount_std": 27.413439,
    "time_since_customer_tx": 18128.0,
    "customer_amount_deviation": 14.898053,
    "customer_amount_ratio": 1.426,
    "customer_tx_count_1h": 0,
    "customer_tx_count_24h": 5,
    "customer_amount_sum_24h": 148.82,
    "terminal_tx_count": 120,
    "terminal_avg_amount": 63.78775,
    "terminal_max_amount": 408.55,
    "terminal_amount_std": 50.07052,
    "terminal_fraud_count": 1,
    "terminal_fraud_rate": 0.008333,
    "terminal_tx_count_1h": 0,
    "terminal_tx_count_24h": 3,
    "terminal_amount_sum_24h": 166.29,
}

HIGH_RISK_TRANSACTION = {
    "TX_AMOUNT": 500.0,
    "hour_of_day": 23,
    "day_of_week": 1,
    "is_weekend": 0,
    "customer_tx_count": 100,
    "customer_avg_amount": 50.0,
    "customer_max_amount": 250.0,
    "customer_amount_std": 40.0,
    "time_since_customer_tx": 3600.0,
    "customer_amount_deviation": 450.0,
    "customer_amount_ratio": 10.0,
    "customer_tx_count_1h": 3,
    "customer_tx_count_24h": 12,
    "customer_amount_sum_24h": 800.0,
    "terminal_tx_count": 200,
    "terminal_avg_amount": 75.0,
    "terminal_max_amount": 500.0,
    "terminal_amount_std": 60.0,
    "terminal_fraud_count": 25,
    "terminal_fraud_rate": 0.125,
    "terminal_tx_count_1h": 5,
    "terminal_tx_count_24h": 20,
    "terminal_amount_sum_24h": 1500.0,
}


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Fraud Risk Intelligence Platform API is running"
    }


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_model_info_endpoint() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == predictor.metadata["model_name"]
    assert data["model_type"] == predictor.metadata["model_type"]
    assert data["decision_threshold"] == predictor.decision_threshold
    assert data["feature_count"] == len(predictor.required_features)
    assert data["features"] == predictor.required_features


def test_single_prediction() -> None:
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200
    data = response.json()

    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert 0.0 <= data["fraud_risk_score"] <= 100.0
    assert data["predicted_fraud"] in [0, 1]
    assert data["decision_threshold"] == predictor.decision_threshold
    assert data["risk_level"] in [
        "minimal",
        "low",
        "medium",
        "high",
        "critical",
    ]


def test_invalid_single_prediction() -> None:
    invalid_transaction = {
        "TX_AMOUNT": -100,
        "hour_of_day": 25,
        "day_of_week": 8,
        "is_weekend": 3,
    }

    response = client.post(
        "/predict",
        json=invalid_transaction,
    )

    assert response.status_code == 422


def test_batch_prediction() -> None:
    response = client.post(
        "/predict/batch",
        json={
            "transactions": [
                VALID_TRANSACTION,
                HIGH_RISK_TRANSACTION,
            ]
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total_transactions"] == 2
    assert data["decision_threshold"] == predictor.decision_threshold
    assert len(data["predictions"]) == 2
    assert 0 <= data["predicted_fraud_count"] <= 2

    assert (
        0.0
        <= data["average_fraud_probability"]
        <= 1.0
    )

    for prediction in data["predictions"]:
        assert (
            0.0
            <= prediction["fraud_probability"]
            <= 1.0
        )
        assert (
            0.0
            <= prediction["fraud_risk_score"]
            <= 100.0
        )
        assert prediction["predicted_fraud"] in [0, 1]
        assert (
            prediction["decision_threshold"]
            == predictor.decision_threshold
        )


def test_empty_batch_prediction() -> None:
    response = client.post(
        "/predict/batch",
        json={
            "transactions": []
        },
    )

    assert response.status_code == 422
