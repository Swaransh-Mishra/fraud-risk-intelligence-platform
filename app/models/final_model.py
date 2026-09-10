from __future__ import annotations

from typing import Any


def get_final_model_metadata(
    model_name: str,
    model_type: str,
    decision_threshold: float,
    feature_count: int,
    ensemble_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Build metadata for a finalized production model.

    Production model identity, threshold, feature count,
    and optional ensemble weights are supplied by the
    finalized model-training workflow.
    """
    metadata: dict[str, Any] = {
        "model_name": model_name,
        "model_type": model_type,
        "decision_threshold": decision_threshold,
        "feature_count": feature_count,
    }

    if ensemble_weights is not None:
        metadata["ensemble_weights"] = ensemble_weights

    return metadata

