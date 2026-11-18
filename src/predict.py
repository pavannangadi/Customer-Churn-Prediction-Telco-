"""Predict churn for new customer records using the saved pipeline.

Usage (from the project root):
    python -m src.predict                      # runs on the built-in example
    python -m src.predict path/to/customers.csv
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

EXAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 140.7,
}


def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Create it first with: python -m src.train")
    return joblib.load(MODEL_PATH)


def predict(df: pd.DataFrame, pipeline=None) -> pd.DataFrame:
    """Return churn probability and a 0/1 flag for each row of `df`."""
    pipeline = pipeline or load_pipeline()
    expected = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["feature_columns"]

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    X = df[expected].copy()  # also fixes column order; ignores extras like customerID
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")  # blank -> NaN

    out = df.copy()
    out["churn_probability"] = pipeline.predict_proba(X)[:, 1].round(4)
    out["churn_predicted"] = pipeline.predict(X)
    return out


def main() -> None:
    if len(sys.argv) > 1:
        data = pd.read_csv(sys.argv[1])
    else:
        data = pd.DataFrame([EXAMPLE_CUSTOMER])
    result = predict(data)
    cols = [c for c in ["customerID", "churn_probability", "churn_predicted"]
            if c in result.columns]
    print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()