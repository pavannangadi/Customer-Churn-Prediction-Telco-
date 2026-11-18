"""Model definitions and the reproducible training script for the Telco churn project.

Run from the project root:
    python -m src.train
"""

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import evaluate_model
from src.preprocessing import (RANDOM_STATE, build_preprocessor, clean_data,
                               get_column_types, get_train_test_split, load_data)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FINAL_MODEL_NAME = "Logistic Regression (balanced)"


def get_models(numeric_cols, categorical_cols) -> dict:
    """Return {name: unfitted Pipeline}. Each gets its OWN preprocessor instance."""
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE),
        "Decision Tree (balanced)": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
    }
    return {
        name: Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", est),
        ])
        for name, est in estimators.items()
    }


def main() -> None:
    """Clean data, split, fit the final pipeline on TRAIN only, and save it."""
    df = clean_data(load_data(str(DATA_PATH)))
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    numeric_cols, categorical_cols = get_column_types(X_train)

    pipeline = get_models(numeric_cols, categorical_cols)[FINAL_MODEL_NAME]
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(FINAL_MODEL_NAME, pipeline,
                             X_train, y_train, X_test, y_test)
    metrics = {k: (v if k == "model" else round(float(v), 4))
               for k, v in metrics.items()}

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "model_name": FINAL_MODEL_NAME,
        "random_state": RANDOM_STATE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": X_train.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "test_metrics": metrics,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved pipeline -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()