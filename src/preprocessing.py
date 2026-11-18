"""Data loading and cleaning helpers for the Telco churn project."""

import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"


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