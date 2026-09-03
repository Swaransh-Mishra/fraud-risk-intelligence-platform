from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from app.core import (
    DriftDetectionError,
    get_logger,
)


logger = get_logger(__name__)


DEFAULT_PSI_THRESHOLD = 0.2

DEFAULT_KS_PVALUE_THRESHOLD = 0.05

DEFAULT_NUM_BINS = 10


def calculate_population_stability_index(
    reference_values: pd.Series,
    current_values: pd.Series,
    num_bins: int = DEFAULT_NUM_BINS,
) -> float:
    """
    Calculate Population Stability Index between
    reference and current numerical distributions.
    """

    try:
        reference = (
            pd.Series(reference_values)
            .dropna()
            .astype(float)
        )

        current = (
            pd.Series(current_values)
            .dropna()
            .astype(float)
        )

        if reference.empty or current.empty:
            return 0.0

        quantiles = np.linspace(
            0.0,
            1.0,
            num_bins + 1,
        )

        bin_edges = np.unique(
            np.quantile(
                reference,
                quantiles,
            )
        )

        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        reference_counts = np.histogram(
            reference,
            bins=bin_edges,
        )[0]

        current_counts = np.histogram(
            current,
            bins=bin_edges,
        )[0]

        reference_distribution = (
            reference_counts
            / reference_counts.sum()
        )

        current_distribution = (
            current_counts
            / current_counts.sum()
        )

        epsilon = 1e-6

        reference_distribution = np.clip(
            reference_distribution,
            epsilon,
            None,
        )

        current_distribution = np.clip(
            current_distribution,
            epsilon,
            None,
        )

        psi = np.sum(
            (
                current_distribution
                - reference_distribution
            )
            * np.log(
                current_distribution
                / reference_distribution
            )
        )

        return float(psi)

    except Exception as error:
        logger.exception(
            "Population Stability Index calculation failed."
        )

        raise DriftDetectionError(
            "Failed to calculate Population "
            "Stability Index."
        ) from error


def calculate_feature_drift(
    reference_values: pd.Series,
    current_values: pd.Series,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_pvalue_threshold: float = (
        DEFAULT_KS_PVALUE_THRESHOLD
    ),
) -> dict[str, Any]:
    """
    Calculate drift metrics for one numerical feature.
    """

    try:
        reference = (
            pd.Series(reference_values)
            .dropna()
            .astype(float)
        )

        current = (
            pd.Series(current_values)
            .dropna()
            .astype(float)
        )

        if reference.empty or current.empty:
            return {
                "psi": 0.0,
                "ks_statistic": 0.0,
                "ks_pvalue": 1.0,
                "drift_detected": False,
            }

        psi = calculate_population_stability_index(
            reference_values=reference,
            current_values=current,
        )

        ks_result = ks_2samp(
            reference,
            current,
        )

        ks_statistic = float(
            ks_result.statistic
        )

        ks_pvalue = float(
            ks_result.pvalue
        )

        drift_detected = (
            psi >= psi_threshold
            or ks_pvalue < ks_pvalue_threshold
        )

        return {
            "psi": round(
                psi,
                6,
            ),
            "ks_statistic": round(
                ks_statistic,
                6,
            ),
            "ks_pvalue": round(
                ks_pvalue,
                6,
            ),
            "drift_detected": bool(
                drift_detected
            ),
        }

    except DriftDetectionError:
        raise

    except Exception as error:
        logger.exception(
            "Feature drift calculation failed."
        )

        raise DriftDetectionError(
            "Failed to calculate feature drift."
        ) from error


def detect_dataset_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: list[str] | None = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_pvalue_threshold: float = (
        DEFAULT_KS_PVALUE_THRESHOLD
    ),
) -> dict[str, Any]:
    """
    Compare reference and current datasets and
    return a feature-level drift report.
    """

    try:
        if reference_data.empty:
            raise ValueError(
                "Reference dataset cannot be empty."
            )

        if current_data.empty:
            raise ValueError(
                "Current dataset cannot be empty."
            )

        if features is None:
            features = [
                column
                for column
                in reference_data.columns
                if column in current_data.columns
            ]

        if not features:
            raise ValueError(
                "No common features are available "
                "for drift detection."
            )

        missing_reference_features = [
            feature
            for feature in features
            if feature
            not in reference_data.columns
        ]

        if missing_reference_features:
            raise ValueError(
                "Features missing from reference data: "
                f"{missing_reference_features}"
            )

        missing_current_features = [
            feature
            for feature in features
            if feature
            not in current_data.columns
        ]

        if missing_current_features:
            raise ValueError(
                "Features missing from current data: "
                f"{missing_current_features}"
            )

        feature_results = {}

        for feature in features:
            feature_results[feature] = (
                calculate_feature_drift(
                    reference_values=(
                        reference_data[feature]
                    ),
                    current_values=(
                        current_data[feature]
                    ),
                    psi_threshold=psi_threshold,
                    ks_pvalue_threshold=(
                        ks_pvalue_threshold
                    ),
                )
            )

        drifted_features = [
            feature
            for feature, result
            in feature_results.items()
            if result["drift_detected"]
        ]

        total_features = len(
            feature_results
        )

        drifted_feature_count = len(
            drifted_features
        )

        drifted_feature_rate = (
            drifted_feature_count
            / total_features
            if total_features > 0
            else 0.0
        )

        report = {
            "reference_records": int(
                len(reference_data)
            ),
            "current_records": int(
                len(current_data)
            ),
            "total_features_checked": int(
                total_features
            ),
            "drifted_feature_count": int(
                drifted_feature_count
            ),
            "drifted_feature_rate": round(
                float(
                    drifted_feature_rate
                ),
                6,
            ),
            "drift_detected": bool(
                drifted_feature_count > 0
            ),
            "drifted_features": (
                drifted_features
            ),
            "feature_results": (
                feature_results
            ),
        }

        logger.info(
            "Dataset drift detection completed | "
            "reference_records=%s | "
            "current_records=%s | "
            "total_features_checked=%s | "
            "drifted_feature_count=%s",
            report["reference_records"],
            report["current_records"],
            report["total_features_checked"],
            report["drifted_feature_count"],
        )

        return report

    except ValueError:
        raise

    except DriftDetectionError:
        raise

    except Exception as error:
        logger.exception(
            "Dataset drift detection failed."
        )

        raise DriftDetectionError(
            "Failed to detect dataset drift."
        ) from error