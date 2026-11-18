"""Model definitions for the Telco churn project."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import RANDOM_STATE, build_preprocessor


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