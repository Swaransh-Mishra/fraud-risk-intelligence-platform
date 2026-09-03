from __future__ import annotations

import pandas as pd


def build_model_comparison(
    model_results: dict[str, dict],
) -> pd.DataFrame:
    """
    Build a clean comparison table for all fraud detection models.

    Each model result should contain metrics returned by
    evaluate_model().
    """

    rows = []

    for model_name, results in model_results.items():
        rows.append(
            {
                "model": model_name,
                "roc_auc": results["roc_auc"],
                "pr_auc": results["pr_auc"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1_score": results["f1_score"],
                "accuracy": results["accuracy"],
                "true_positives": results["true_positives"],
                "false_positives": results["false_positives"],
                "false_negatives": results["false_negatives"],
                "true_negatives": results["true_negatives"],
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        by="pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    return comparison