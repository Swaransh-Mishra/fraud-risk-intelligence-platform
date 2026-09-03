
from typing import Any

from catboost import CatBoostClassifier


FINAL_MODEL_NAME = "CatBoost Fraud Risk Classifier"
FINAL_MODEL_TYPE = "catboost_classifier"
FINAL_DECISION_THRESHOLD = 0.65


def build_final_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def get_final_model_metadata() -> dict[str, Any]:
    return {
        "model_name": FINAL_MODEL_NAME,
        "model_type": FINAL_MODEL_TYPE,
        "decision_threshold": FINAL_DECISION_THRESHOLD,
        "feature_count": 23,
    }

