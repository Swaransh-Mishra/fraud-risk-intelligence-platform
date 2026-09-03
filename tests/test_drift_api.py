from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


REFERENCE_DATA = [
    {
        "amount": 10,
        "count": 1,
    },
    {
        "amount": 11,
        "count": 2,
    },
    {
        "amount": 12,
        "count": 3,
    },
    {
        "amount": 13,
        "count": 4,
    },
    {
        "amount": 14,
        "count": 5,
    },
    {
        "amount": 15,
        "count": 6,
    },
    {
        "amount": 16,
        "count": 7,
    },
    {
        "amount": 17,
        "count": 8,
    },
    {
        "amount": 18,
        "count": 9,
    },
    {
        "amount": 19,
        "count": 10,
    },
]


SHIFTED_CURRENT_DATA = [
    {
        "amount": 100,
        "count": 1,
    },
    {
        "amount": 110,
        "count": 2,
    },
    {
        "amount": 120,
        "count": 3,
    },
    {
        "amount": 130,
        "count": 4,
    },
    {
        "amount": 140,
        "count": 5,
    },
    {
        "amount": 150,
        "count": 6,
    },
    {
        "amount": 160,
        "count": 7,
    },
    {
        "amount": 170,
        "count": 8,
    },
    {
        "amount": 180,
        "count": 9,
    },
    {
        "amount": 190,
        "count": 10,
    },
]


IDENTICAL_CURRENT_DATA = [
    {
        "amount": 10,
        "count": 1,
    },
    {
        "amount": 11,
        "count": 2,
    },
    {
        "amount": 12,
        "count": 3,
    },
    {
        "amount": 13,
        "count": 4,
    },
    {
        "amount": 14,
        "count": 5,
    },
    {
        "amount": 15,
        "count": 6,
    },
    {
        "amount": 16,
        "count": 7,
    },
    {
        "amount": 17,
        "count": 8,
    },
    {
        "amount": 18,
        "count": 9,
    },
    {
        "amount": 19,
        "count": 10,
    },
]


def test_drift_endpoint_detects_shift():
    """
    Verify that the API detects a meaningful
    distribution shift.
    """

    response = client.post(
        "/monitoring/drift",
        json={
            "reference_data": REFERENCE_DATA,
            "current_data": SHIFTED_CURRENT_DATA,
            "features": [
                "amount",
                "count",
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["drift_detected"] is True
    assert "amount" in data["drifted_features"]
    assert "count" not in data["drifted_features"]


def test_drift_endpoint_returns_feature_results():
    """
    Verify that feature-level drift metrics are
    returned by the API.
    """

    response = client.post(
        "/monitoring/drift",
        json={
            "reference_data": REFERENCE_DATA,
            "current_data": SHIFTED_CURRENT_DATA,
            "features": [
                "amount",
                "count",
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "feature_results" in data
    assert "amount" in data["feature_results"]
    assert "count" in data["feature_results"]

    amount_result = (
        data["feature_results"]["amount"]
    )

    assert amount_result["drift_detected"] is True
    assert amount_result["psi"] > 0
    assert amount_result["ks_statistic"] > 0


def test_drift_endpoint_detects_no_drift():
    """
    Verify that identical datasets do not trigger
    drift detection.
    """

    response = client.post(
        "/monitoring/drift",
        json={
            "reference_data": REFERENCE_DATA,
            "current_data": IDENTICAL_CURRENT_DATA,
            "features": [
                "amount",
                "count",
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["drift_detected"] is False
    assert data["drifted_features"] == []


def test_drift_endpoint_with_selected_feature():
    """
    Verify that only selected features are evaluated.
    """

    response = client.post(
        "/monitoring/drift",
        json={
            "reference_data": REFERENCE_DATA,
            "current_data": SHIFTED_CURRENT_DATA,
            "features": [
                "count",
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["drift_detected"] is False

    assert data["evaluated_features"] == [
        "count"
    ]

    assert list(
        data["feature_results"].keys()
    ) == [
        "count"
    ]


def test_drift_endpoint_rejects_missing_feature():
    """
    Verify that the API rejects a requested feature
    that does not exist in the datasets.
    """

    response = client.post(
        "/monitoring/drift",
        json={
            "reference_data": REFERENCE_DATA,
            "current_data": SHIFTED_CURRENT_DATA,
            "features": [
                "missing_feature",
            ],
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "detail" in data
    assert "missing_feature" in data["detail"]


def test_drift_endpoint_rejects_empty_datasets():
    """
    Verify that empty datasets are rejected
    by request validation.
    """

    response = client.post(
        "/monitoring/drift",
        json={
            "reference_data": [],
            "current_data": [],
        },
    )

    assert response.status_code == 422