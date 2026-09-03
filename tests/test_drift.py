import pandas as pd

from app.monitoring.drift import (
    calculate_feature_drift,
    calculate_population_stability_index,
    detect_dataset_drift,
)


def test_psi_for_identical_distributions():
    """
    PSI should be near zero when the reference and
    current distributions are identical.
    """

    reference = pd.Series(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
        ]
    )

    current = reference.copy()

    psi = calculate_population_stability_index(
        reference_values=reference,
        current_values=current,
    )

    assert psi < 0.01


def test_psi_detects_distribution_shift():
    """
    PSI should increase when the current distribution
    differs substantially from the reference.
    """

    reference = pd.Series(
        [
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ]
    )

    current = pd.Series(
        [
            100,
            110,
            120,
            130,
            140,
            150,
            160,
            170,
            180,
            190,
        ]
    )

    psi = calculate_population_stability_index(
        reference_values=reference,
        current_values=current,
    )

    assert psi > 0.2


def test_feature_drift_for_identical_data():
    """
    Identical feature distributions should not
    be classified as drifted.
    """

    reference = pd.Series(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
        ]
    )

    current = reference.copy()

    result = calculate_feature_drift(
        reference_values=reference,
        current_values=current,
    )

    assert result["drift_detected"] is False
    assert result["psi"] < 0.01


def test_feature_drift_for_shifted_data():
    """
    A substantially shifted distribution should
    be classified as drifted.
    """

    reference = pd.Series(
        [
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ]
    )

    current = pd.Series(
        [
            100,
            110,
            120,
            130,
            140,
            150,
            160,
            170,
            180,
            190,
        ]
    )

    result = calculate_feature_drift(
        reference_values=reference,
        current_values=current,
    )

    assert result["drift_detected"] is True
    assert result["psi"] >= 0.2


def test_dataset_drift_detection():
    """
    Dataset-level monitoring should identify the
    shifted feature and calculate the correct summary.
    """

    reference_data = pd.DataFrame(
        {
            "amount": [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
            ],
            "count": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
        }
    )

    current_data = pd.DataFrame(
        {
            "amount": [
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
                180,
                190,
            ],
            "count": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
        }
    )

    report = detect_dataset_drift(
        reference_data=reference_data,
        current_data=current_data,
    )

    assert report["drift_detected"] is True

    assert (
        report["total_features_checked"]
        == 2
    )

    assert (
        report["drifted_feature_count"]
        == 1
    )

    assert (
        report["drifted_features"]
        == ["amount"]
    )

    assert (
        report["feature_results"]["amount"]
        ["drift_detected"]
        is True
    )

    assert (
        report["feature_results"]["count"]
        ["drift_detected"]
        is False
    )


def test_dataset_drift_with_selected_features():
    """
    Drift detection should evaluate only the
    explicitly selected features.
    """

    reference_data = pd.DataFrame(
        {
            "amount": [
                10,
                20,
                30,
                40,
            ],
            "count": [
                1,
                2,
                3,
                4,
            ],
        }
    )

    current_data = pd.DataFrame(
        {
            "amount": [
                100,
                200,
                300,
                400,
            ],
            "count": [
                1,
                2,
                3,
                4,
            ],
        }
    )

    report = detect_dataset_drift(
        reference_data=reference_data,
        current_data=current_data,
        features=["count"],
    )

    assert (
        report["total_features_checked"]
        == 1
    )

    assert (
        report["drift_detected"]
        is False
    )

    assert report["drifted_features"] == []


def test_missing_feature_raises_error():
    """
    Requesting a feature that does not exist in
    the current dataset should raise ValueError.
    """

    reference_data = pd.DataFrame(
        {
            "amount": [
                10,
                20,
                30,
            ],
        }
    )

    current_data = pd.DataFrame(
        {
            "different_amount": [
                10,
                20,
                30,
            ],
        }
    )

    try:
        detect_dataset_drift(
            reference_data=reference_data,
            current_data=current_data,
            features=["amount"],
        )

    except ValueError as error:
        assert (
            "missing from current data"
            in str(error)
        )

    else:
        raise AssertionError(
            "Expected ValueError was not raised"
        )


def test_empty_data_does_not_crash():
    """
    Empty feature data should return a stable
    non-drifted result.
    """

    reference = pd.Series(
        dtype=float
    )

    current = pd.Series(
        dtype=float
    )

    result = calculate_feature_drift(
        reference_values=reference,
        current_values=current,
    )

    assert result["psi"] == 0.0
    assert result["ks_statistic"] == 0.0
    assert result["ks_pvalue"] == 1.0

    assert (
        result["drift_detected"]
        is False
    )