"""Data loading and cleaning helpers for the Telco churn project."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"

RANDOM_STATE = 42


def split_features_target(df: pd.DataFrame):
    """Separate the feature matrix X from the target y."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """80/20 stratified split so both sets keep the same churn ratio."""
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def get_column_types(X: pd.DataFrame):
    """Return (numeric_columns, categorical_columns).

    SeniorCitizen is stored as 0/1 but is really a yes/no flag; it is kept
    numeric, which is fine for these models.
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Unfitted preprocessing: impute + scale numerics, impute + one-hot categoricals."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def load_data(path: str) -> pd.DataFrame:
    """Read the raw Telco churn CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataframe.

    Steps:
      1. Convert TotalCharges from text to numeric (blank strings -> NaN).
      2. Fill those NaNs with 0, but only if every affected row has tenure == 0
         (brand-new customers not yet billed). Otherwise raise an error.
      3. Drop the customerID identifier.
      4. Encode Churn as 0 (No) / 1 (Yes).
    """
    df = df.copy()

    # 1. TotalCharges: text -> numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")

    # 2. Handle the blanks with a verified, domain-based rule
    missing_mask = df["TotalCharges"].isna()
    if not (df.loc[missing_mask, "tenure"] == 0).all():
        raise ValueError(
            "Some rows with missing TotalCharges have tenure > 0; "
            "the 'tenure == 0 -> TotalCharges = 0' rule does not apply."
        )
    df.loc[missing_mask, "TotalCharges"] = 0.0

    # 3. Drop identifier
    df = df.drop(columns=[ID_COLUMN])

    # 4. Encode target
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    if df[TARGET].isna().any():
        raise ValueError("Unexpected values found in Churn column.")

    return df